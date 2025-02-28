from fastapi import FastAPI, Form, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Union
import google.generativeai as genai
import json
import re
import asyncio
import aiohttp
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="ATS Resume Analyzer API",
    description="API for analyzing resumes against job descriptions using Gemini AI",
    version="1.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class ResumeInput(BaseModel):
    url: HttpUrl = Field(..., description="URL to the resume image file (JPG format)")
    filename: Optional[str] = Field(None, description="Optional custom filename")
    user_id: str = Field(..., description="User ID associated with this resume")


class ATSRequest(BaseModel):
    job_description: str = Field(..., description="The job description to match against")
    resumes: List[ResumeInput] = Field(..., description="List of resume URLs to analyze with user IDs")


class UserResult(BaseModel):
    user_id: str = Field(..., description="User ID")
    score: float = Field(..., description="Match score percentage (0-100)")
    sgAnalysis: str = Field(..., description="Analysis of skills gap between resume and job requirements")


class ATSResponse(BaseModel):
    results: List[UserResult] = Field(..., description="Analysis results for each user")


# Dependency for API key
def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY not configured on server"
        )
    return api_key


# Configure Gemini API
def configure_genai(api_key: str):
    genai.configure(api_key=api_key)


# Prepare prompt for Gemini (for direct image processing)
def prepare_image_prompt(resume_url, job_description):
    prompt_template = """
    You are looking at a resume image. First, extract all the text content from the resume.

    Then act as an expert ATS (Applicant Tracking System) specialist with deep expertise in:
    - Technical fields
    - Software engineering
    - Data science
    - Data analysis
    - Big data engineering

    Evaluate the resume against the job description. Consider that the job market
    is highly competitive. Provide detailed feedback for resume improvement.

    Job Description:
    {job_description}

    Provide a response in the following JSON format ONLY, with no additional text:
    {{
        "JD Match": "percentage between 0-100 (just provide the number, e.g., 75)",
        "MissingKeywords": ["keyword1", "keyword2", ...],
        "Profile Summary": "detailed analysis of the match and specific improvement suggestions",
        "Skills Gap Analysis": "comprehensive analysis of the specific skills gap between the candidate's resume and the job requirements, including technical skills, tools, methodologies, and experience levels that are missing or insufficient"
    }}
    """

    return prompt_template.format(
        job_description=job_description.strip()
    )


# Get response from Gemini API with image URL
async def get_gemini_image_response(prompt, image_url, api_key, max_retries=3, retry_delay=2):
    configure_genai(api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, image_url])
            logger.info(f"Received response from Gemini Vision API (attempt {attempt + 1})")

            if not response or not response.text:
                logger.warning(f"Empty response from Gemini API (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Empty response received from Gemini API after retries"
                    )

            try:
                # Log the raw response for debugging
                logger.info(f"Raw Gemini response: {response.text[:500]}...")

                # Try to parse the JSON response
                json_pattern = r'\{.*\}'
                match = re.search(json_pattern, response.text, re.DOTALL)
                if match:
                    try:
                        extracted_json = match.group()
                        logger.info(f"Extracted JSON: {extracted_json[:500]}...")
                        response_json = json.loads(extracted_json)

                        # Check for required fields
                        required_fields = ["JD Match", "MissingKeywords", "Profile Summary", "Skills Gap Analysis"]
                        for field in required_fields:
                            if field not in response_json:
                                logger.warning(f"Missing required field in Gemini response: {field}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay)
                                    continue
                                else:
                                    raise ValueError(f"Missing required field: {field}")

                        return response_json
                    except json.JSONDecodeError:
                        if attempt < max_retries - 1:
                            logger.warning("JSON decode error in extracted text, retrying...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            raise HTTPException(
                                status_code=status.HTTP_502_BAD_GATEWAY,
                                detail="Could not parse JSON content from Gemini response"
                            )
                else:
                    if attempt < max_retries - 1:
                        logger.warning("Could not extract JSON pattern, retrying...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail="Could not extract valid JSON from Gemini response"
                        )

            except Exception as e:
                logger.error(f"Error processing Gemini response: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Error processing Gemini response after retries: {str(e)}"
                    )

        except Exception as e:
            logger.error(f"Error with Gemini API: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            else:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gemini API error after retries: {str(e)}"
                )


# Extract score from various formats
def extract_score(match_percentage):
    logger.info(f"Extracting score from: {match_percentage}")

    # Handle if it's already a number
    if isinstance(match_percentage, (int, float)):
        return float(match_percentage)

    # Handle if it's a string containing just a number
    if isinstance(match_percentage, str):
        try:
            # Try direct conversion first
            return float(match_percentage.strip())
        except ValueError:
            # If that fails, try to extract a number with regex
            pass

    # Use regex to find a number pattern in various formats
    patterns = [
        r'(\d+\.?\d*)',  # Match numbers like 75 or 75.5
        r'(\d+\.?\d*)%',  # Match percentage format like 75% or 75.5%
        r'(\d+\.?\d*)\s*percent',  # Match "75 percent" or "75.5 percent"
        r'(\d+\.?\d*)\s*out of\s*100',  # Match "75 out of 100"
    ]

    for pattern in patterns:
        match = re.search(pattern, str(match_percentage))
        if match:
            try:
                extracted_value = float(match.group(1))
                logger.info(f"Successfully extracted score: {extracted_value}")
                return extracted_value
            except ValueError:
                continue

    # If all extraction methods fail, log and return 0
    logger.warning(f"Could not extract score from: {match_percentage}, defaulting to 0")
    return 0.0


# Process resume from URL directly with Gemini Vision
async def process_resume_url(resume_input: ResumeInput, job_description: str, api_key: str):
    try:
        url = str(resume_input.url)
        user_id = resume_input.user_id

        logger.info(f"Processing resume for user: {user_id} from URL: {url}")

        # Generate prompt for image analysis
        prompt = prepare_image_prompt(url, job_description)

        # Send image URL directly to Gemini Vision API
        response = await get_gemini_image_response(prompt, url, api_key)
        logger.info(f"Received analysis for user: {user_id}")

        # Extract match percentage - with improved handling
        match_percentage = response.get("JD Match", "0")
        score = extract_score(match_percentage)

        # Create result with user_id
        result = {
            "user_id": user_id,
            "score": score,
            "sgAnalysis": response.get("Skills Gap Analysis", "Skills gap analysis not available")
        }

        logger.info(f"Analysis complete for user: {user_id}, score: {score}")
        return result

    except HTTPException as e:
        # Re-raise HTTP exceptions as is
        logger.error(f"HTTP exception while processing resume for user {user_id}: {str(e)}")
        raise e
    except Exception as e:
        # Wrap other exceptions
        logger.error(f"Error processing resume for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing resume for user {user_id}: {str(e)}"
        )


# Process multiple resumes in parallel with rate limiting
async def process_resumes(resumes: List[ResumeInput], job_description: str, api_key: str):
    # Use a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(3)  # Process up to 3 resumes concurrently

    async def process_with_semaphore(resume):
        async with semaphore:
            try:
                return await process_resume_url(resume, job_description, api_key)
            except Exception as e:
                logger.error(f"Error in process_with_semaphore for user {resume.user_id}: {str(e)}")
                return {
                    "user_id": resume.user_id,
                    "score": 0.0,
                    "sgAnalysis": "Could not analyze skills gap due to processing error."
                }

    # Create tasks for all resumes
    tasks = [process_with_semaphore(resume) for resume in resumes]

    # Execute all tasks
    results = await asyncio.gather(*tasks)

    # Sort results by score
    ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)

    return ranked_results


# Endpoints
@app.post("/analyze", response_model=ATSResponse,
          summary="Analyze resumes against a job description",
          description="Provide resume image URLs with user IDs and a job description to get ATS analysis")
async def analyze_resumes_endpoint(
        request: ATSRequest,
        api_key: str = Depends(get_api_key)
):
    if not request.resumes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resume URLs provided"
        )

    logger.info(f"Received analyze request with {len(request.resumes)} resumes")
    results = await process_resumes(request.resumes, request.job_description, api_key)
    logger.info(f"Analysis complete for {len(results)} resumes")
    return {"results": results}


# Form-based endpoint for backward compatibility
@app.post("/analyze-form", response_model=ATSResponse,
          summary="Analyze resumes against a job description (form-based)",
          description="Provide resume URLs with user IDs and a job description using form data")
async def analyze_resumes_form(
        job_description: str = Form(...),
        resume_urls: str = Form(...),  # Comma-separated URLs
        user_ids: str = Form(...),  # Comma-separated user IDs
        api_key: str = Depends(get_api_key)
):
    # Parse resume URLs and user IDs from form data
    urls = [url.strip() for url in resume_urls.split(",") if url.strip()]
    ids = [uid.strip() for uid in user_ids.split(",") if uid.strip()]

    if not urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resume URLs provided"
        )

    if len(urls) != len(ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The number of resume URLs must match the number of user IDs"
        )

    logger.info(f"Received form-based analyze request with {len(urls)} resumes")

    # Convert to ResumeInput objects with user IDs
    resume_inputs = [ResumeInput(url=url, user_id=uid) for url, uid in zip(urls, ids)]

    results = await process_resumes(resume_inputs, job_description, api_key)
    logger.info(f"Form-based analysis complete for {len(results)} resumes")
    return {"results": results}


# Health check endpoint
@app.get("/health",
         summary="Health check endpoint",
         description="Check if the API is running")
def health_check():
    return {"status": "healthy"}


# Main driver
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting ATS Resume Analyzer API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field
# from typing import List, Optional
# import google.generativeai as genai
# import PyPDF2 as pdf
# import json
# import pandas as pd
# import io
# import re
# import time
# import os
# from dotenv import load_dotenv
#
# # Load environment variables
# load_dotenv()
#
# app = FastAPI(
#     title="ATS Resume Analyzer API",
#     description="API for analyzing resumes against job descriptions using Gemini AI",
#     version="1.0.0"
# )
#
# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Modify in production to specific origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# # Models
# class ATSRequest(BaseModel):
#     job_description: str = Field(..., description="The job description to match against")
#
#
# class MissingKeywords(BaseModel):
#     keywords: List[str] = Field(default_factory=list, description="Keywords missing from the resume")
#
#
# class ResumeResult(BaseModel):
#     resume_name: str = Field(..., description="Filename of the resume")
#     score: float = Field(..., description="Match score percentage (0-100)")
#     missing_keywords: List[str] = Field(default_factory=list, description="Keywords missing from the resume")
#     profile_summary: str = Field(..., description="Detailed analysis and suggestions")
#
#
# class ATSResponse(BaseModel):
#     results: List[ResumeResult] = Field(..., description="Analysis results for each resume")
#
#
# # Dependency for API key
# def get_api_key():
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="GEMINI_API_KEY not configured on server"
#         )
#     return api_key
#
#
# # Configure Gemini API
# def configure_genai(api_key: str):
#     genai.configure(api_key=api_key)
#
#
# # Extract text from PDF
# def extract_pdf_text(pdf_bytes):
#     try:
#         reader = pdf.PdfReader(io.BytesIO(pdf_bytes))
#         text = []
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text.append(page_text)
#         return " ".join(text)
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#             detail=f"PDF extraction error: {str(e)}"
#         )
#
#
# # Prepare prompt for Gemini
# def prepare_prompt(resume_text, job_description):
#     prompt_template = """
#     Act as an expert ATS (Applicant Tracking System) specialist with deep expertise in:
#     - Technical fields
#     - Software engineering
#     - Data science
#     - Data analysis
#     - Big data engineering
#
#     Evaluate the following resume against the job description. Consider that the job market
#     is highly competitive. Provide detailed feedback for resume improvement.
#
#     Resume:
#     {resume_text}
#
#     Job Description:
#     {job_description}
#
#     Provide a response in the following JSON format ONLY:
#     {{
#         "JD Match": "percentage between 0-100",
#         "MissingKeywords": ["keyword1", "keyword2", ...],
#         "Profile Summary": "detailed analysis of the match and specific improvement suggestions"
#     }}
#     """
#
#     return prompt_template.format(
#         resume_text=resume_text.strip(),
#         job_description=job_description.strip()
#     )
#
#
# # Get response from Gemini API
# def get_gemini_response(prompt, api_key):
#     try:
#         configure_genai(api_key)
#         model = genai.GenerativeModel('gemini-1.5-pro')
#         response = model.generate_content(prompt)
#
#         if not response or not response.text:
#             raise HTTPException(
#                 status_code=status.HTTP_502_BAD_GATEWAY,
#                 detail="Empty response received from Gemini API"
#             )
#
#         try:
#             response_json = json.loads(response.text)
#
#             required_fields = ["JD Match", "MissingKeywords", "Profile Summary"]
#             for field in required_fields:
#                 if field not in response_json:
#                     raise ValueError(f"Missing required field: {field}")
#
#             return response_json
#
#         except json.JSONDecodeError:
#             json_pattern = r'\{.*\}'
#             match = re.search(json_pattern, response.text, re.DOTALL)
#             if match:
#                 try:
#                     return json.loads(match.group())
#                 except:
#                     raise HTTPException(
#                         status_code=status.HTTP_502_BAD_GATEWAY,
#                         detail="Could not parse extracted JSON content from Gemini response"
#                     )
#             else:
#                 raise HTTPException(
#                     status_code=status.HTTP_502_BAD_GATEWAY,
#                     detail="Could not extract valid JSON from Gemini response"
#                 )
#
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_502_BAD_GATEWAY,
#             detail=f"Gemini API error: {str(e)}"
#         )
#
#
# # Process resume file
# async def process_resume(file: UploadFile, job_description: str, api_key: str):
#     try:
#         filename = file.filename
#         content = await file.read()
#
#         # Extract text
#         resume_text = extract_pdf_text(content)
#
#         # Generate prompt and get response
#         prompt = prepare_prompt(resume_text, job_description)
#         response = get_gemini_response(prompt, api_key)
#
#         # Extract match percentage
#         match_percentage = response["JD Match"]
#         match = re.search(r'(\d+)', str(match_percentage))
#         score = float(match.group(1)) if match else 0.0
#
#         # Create result
#         result = {
#             "resume_name": filename,
#             "score": score,
#             "missing_keywords": response["MissingKeywords"],
#             "profile_summary": response["Profile Summary"]
#         }
#
#         return result
#
#     except HTTPException as e:
#         # Re-raise HTTP exceptions as is
#         raise e
#     except Exception as e:
#         # Wrap other exceptions
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing resume {file.filename}: {str(e)}"
#         )
#
#
# # Endpoints
# @app.post("/analyze", response_model=ATSResponse,
#           summary="Analyze resumes against a job description",
#           description="Upload multiple resume PDFs and a job description to get ATS analysis")
# async def analyze_resumes(
#         background_tasks: BackgroundTasks,
#         job_description: str = Form(...),
#         resumes: List[UploadFile] = File(...),
#         api_key: str = Depends(get_api_key)
# ):
#     if not resumes:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="No resume files uploaded"
#         )
#
#     results = []
#     for resume in resumes:
#         if not resume.filename.lower().endswith('.pdf'):
#             results.append({
#                 "resume_name": resume.filename,
#                 "score": 0.0,
#                 "missing_keywords": [],
#                 "profile_summary": "Error: File is not a PDF"
#             })
#             continue
#
#         try:
#             result = await process_resume(resume, job_description, api_key)
#             results.append(result)
#             # Add small delay between API calls
#             if resume != resumes[-1]:  # Don't delay after the last resume
#                 await asyncio.sleep(1)
#         except Exception as e:
#             results.append({
#                 "resume_name": resume.filename,
#                 "score": 0.0,
#                 "missing_keywords": [],
#                 "profile_summary": f"Error processing resume: {str(e)}"
#             })
#
#     # Sort results by score
#     ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
#
#     return {"results": ranked_results}
#
#
# # Health check endpoint
# @app.get("/health",
#          summary="Health check endpoint",
#          description="Check if the API is running")
# def health_check():
#     return {"status": "healthy"}
#
#
# # Main driver
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# from fastapi import FastAPI, Form, HTTPException, Depends, status, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field, HttpUrl
# from typing import List, Optional, Union
# import google.generativeai as genai
# import PyPDF2 as pdf
# import json
# import io
# import re
# import asyncio
# import aiohttp
# import os
# import logging
# from dotenv import load_dotenv
#
# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# # Load environment variables
# load_dotenv()
#
# app = FastAPI(
#     title="ATS Resume Analyzer API",
#     description="API for analyzing resumes against job descriptions using Gemini AI",
#     version="1.1.0"
# )
#
# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Modify in production to specific origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# # Models
# class ResumeInput(BaseModel):
#     url: HttpUrl = Field(..., description="URL to the resume PDF file")
#     filename: Optional[str] = Field(None, description="Optional custom filename")
#     user_id: str = Field(..., description="User ID associated with this resume")
#
#
# class ATSRequest(BaseModel):
#     job_description: str = Field(..., description="The job description to match against")
#     resumes: List[ResumeInput] = Field(..., description="List of resume URLs to analyze with user IDs")
#
#
# class UserResult(BaseModel):
#     user_id: str = Field(..., description="User ID")
#     score: float = Field(..., description="Match score percentage (0-100)")
#     sgAnalysis: str = Field(..., description="Analysis of skills gap between resume and job requirements")
#
#
# class ATSResponse(BaseModel):
#     results: List[UserResult] = Field(..., description="Analysis results for each user")
#
#
# # Dependency for API key
# def get_api_key():
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="GEMINI_API_KEY not configured on server"
#         )
#     return api_key
#
#
# # Configure Gemini API
# def configure_genai(api_key: str):
#     genai.configure(api_key=api_key)
#
#
# # Extract text from PDF
# def extract_pdf_text(pdf_bytes):
#     try:
#         reader = pdf.PdfReader(io.BytesIO(pdf_bytes))
#         text = []
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text.append(page_text)
#         return " ".join(text)
#     except Exception as e:
#         logger.error(f"PDF extraction error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#             detail=f"PDF extraction error: {str(e)}"
#         )
#
#
# # Download PDF from URL
# async def download_pdf(url, session):
#     try:
#         async with session.get(url) as response:
#             if response.status != 200:
#                 logger.error(f"Failed to download PDF: {url}, status: {response.status}")
#                 raise HTTPException(
#                     status_code=status.HTTP_404_NOT_FOUND,
#                     detail=f"Failed to download PDF from URL: {url}, status: {response.status}"
#                 )
#             return await response.read()
#     except aiohttp.ClientError as e:
#         logger.error(f"Error downloading PDF: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail=f"Error downloading PDF: {str(e)}"
#         )
#
#
# # Prepare prompt for Gemini
# def prepare_prompt(resume_text, job_description):
#     prompt_template = """
#     Act as an expert ATS (Applicant Tracking System) specialist with deep expertise in:
#     - Technical fields
#     - Software engineering
#     - Data science
#     - Data analysis
#     - Big data engineering
#
#     Evaluate the following resume against the job description. Consider that the job market
#     is highly competitive. Provide detailed feedback for resume improvement.
#
#     Resume:
#     {resume_text}
#
#     Job Description:
#     {job_description}
#
#     Provide a response in the following JSON format ONLY, with no additional text:
#     {{
#         "JD Match": "percentage between 0-100 (just provide the number, e.g., 75)",
#         "MissingKeywords": ["keyword1", "keyword2", ...],
#         "Profile Summary": "detailed analysis of the match and specific improvement suggestions",
#         "Skills Gap Analysis": "comprehensive analysis of the specific skills gap between the candidate's resume and the job requirements, including technical skills, tools, methodologies, and experience levels that are missing or insufficient"
#     }}
#     """
#
#     return prompt_template.format(
#         resume_text=resume_text.strip(),
#         job_description=job_description.strip()
#     )
#
#
# # Get response from Gemini API with retry mechanism
# async def get_gemini_response(prompt, api_key, max_retries=3, retry_delay=2):
#     configure_genai(api_key)
#     model = genai.GenerativeModel('gemini-1.5-pro')
#
#     for attempt in range(max_retries):
#         try:
#             response = model.generate_content(prompt)
#             logger.info(f"Received response from Gemini API (attempt {attempt + 1})")
#
#             if not response or not response.text:
#                 logger.warning(f"Empty response from Gemini API (attempt {attempt + 1})")
#                 if attempt < max_retries - 1:
#                     await asyncio.sleep(retry_delay)
#                     continue
#                 else:
#                     raise HTTPException(
#                         status_code=status.HTTP_502_BAD_GATEWAY,
#                         detail="Empty response received from Gemini API after retries"
#                     )
#
#             try:
#                 # Log the raw response for debugging
#                 logger.info(f"Raw Gemini response: {response.text[:500]}...")
#
#                 # Try to parse the JSON response
#                 response_json = json.loads(response.text)
#
#                 required_fields = ["JD Match", "MissingKeywords", "Profile Summary", "Skills Gap Analysis"]
#                 for field in required_fields:
#                     if field not in response_json:
#                         logger.warning(f"Missing required field in Gemini response: {field}")
#                         raise ValueError(f"Missing required field: {field}")
#
#                 return response_json
#
#             except json.JSONDecodeError:
#                 logger.warning(f"JSON decode error in Gemini response (attempt {attempt + 1})")
#                 json_pattern = r'\{.*\}'
#                 match = re.search(json_pattern, response.text, re.DOTALL)
#                 if match:
#                     try:
#                         extracted_json = match.group()
#                         logger.info(f"Extracted JSON: {extracted_json[:500]}...")
#                         return json.loads(extracted_json)
#                     except:
#                         if attempt < max_retries - 1:
#                             logger.warning("Could not parse extracted JSON, retrying...")
#                             await asyncio.sleep(retry_delay)
#                             continue
#                         else:
#                             raise HTTPException(
#                                 status_code=status.HTTP_502_BAD_GATEWAY,
#                                 detail="Could not parse extracted JSON content from Gemini response"
#                             )
#                 else:
#                     if attempt < max_retries - 1:
#                         logger.warning("Could not extract JSON pattern, retrying...")
#                         await asyncio.sleep(retry_delay)
#                         continue
#                     else:
#                         raise HTTPException(
#                             status_code=status.HTTP_502_BAD_GATEWAY,
#                             detail="Could not extract valid JSON from Gemini response"
#                         )
#
#         except Exception as e:
#             logger.error(f"Error with Gemini API: {str(e)}")
#             if attempt < max_retries - 1:
#                 await asyncio.sleep(retry_delay)
#                 continue
#             else:
#                 raise HTTPException(
#                     status_code=status.HTTP_502_BAD_GATEWAY,
#                     detail=f"Gemini API error after retries: {str(e)}"
#                 )
#
#
# # Extract score from various formats
# def extract_score(match_percentage):
#     logger.info(f"Extracting score from: {match_percentage}")
#
#     # Handle if it's already a number
#     if isinstance(match_percentage, (int, float)):
#         return float(match_percentage)
#
#     # Handle if it's a string containing just a number
#     if isinstance(match_percentage, str):
#         try:
#             # Try direct conversion first
#             return float(match_percentage.strip())
#         except ValueError:
#             # If that fails, try to extract a number with regex
#             pass
#
#     # Use regex to find a number pattern in various formats
#     patterns = [
#         r'(\d+\.?\d*)',  # Match numbers like 75 or 75.5
#         r'(\d+\.?\d*)%',  # Match percentage format like 75% or 75.5%
#         r'(\d+\.?\d*)\s*percent',  # Match "75 percent" or "75.5 percent"
#         r'(\d+\.?\d*)\s*out of\s*100',  # Match "75 out of 100"
#     ]
#
#     for pattern in patterns:
#         match = re.search(pattern, str(match_percentage))
#         if match:
#             try:
#                 extracted_value = float(match.group(1))
#                 logger.info(f"Successfully extracted score: {extracted_value}")
#                 return extracted_value
#             except ValueError:
#                 continue
#
#     # If all extraction methods fail, log and return 0
#     logger.warning(f"Could not extract score from: {match_percentage}, defaulting to 0")
#     return 0.0
#
#
# # Process resume from URL
# async def process_resume_url(resume_input: ResumeInput, job_description: str, api_key: str, session):
#     try:
#         url = str(resume_input.url)
#         filename = resume_input.filename or url.split("/")[-1]
#         user_id = resume_input.user_id
#
#         if not filename.lower().endswith('.pdf'):
#             filename += '.pdf'
#
#         logger.info(f"Processing resume for user: {user_id} from URL: {url}")
#
#         # Download the PDF
#         pdf_bytes = await download_pdf(url, session)
#         logger.info(f"Downloaded PDF for user: {user_id}, size: {len(pdf_bytes)} bytes")
#
#         # Extract text
#         resume_text = extract_pdf_text(pdf_bytes)
#         logger.info(f"Extracted text from PDF for user: {user_id}, length: {len(resume_text)} characters")
#
#         # Generate prompt and get response
#         prompt = prepare_prompt(resume_text, job_description)
#         response = await get_gemini_response(prompt, api_key)
#         logger.info(f"Received analysis for user: {user_id}")
#
#         # Extract match percentage - with improved handling
#         match_percentage = response.get("JD Match", "0")
#         score = extract_score(match_percentage)
#
#         # Create result with user_id
#         result = {
#             "user_id": user_id,
#             "score": score,
#             "sgAnalysis": response.get("Skills Gap Analysis", "Skills gap analysis not available")
#         }
#
#         logger.info(f"Analysis complete for user: {user_id}, score: {score}")
#         return result
#
#     except HTTPException as e:
#         # Re-raise HTTP exceptions as is
#         logger.error(f"HTTP exception while processing resume for user {user_id}: {str(e)}")
#         raise e
#     except Exception as e:
#         # Wrap other exceptions
#         logger.error(f"Error processing resume for user {user_id}: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing resume for user {user_id}: {str(e)}"
#         )
#
#
# # Process multiple resumes in parallel with rate limiting
# async def process_resumes(resumes: List[ResumeInput], job_description: str, api_key: str):
#     async with aiohttp.ClientSession() as session:
#         # Use a semaphore to limit concurrent requests
#         semaphore = asyncio.Semaphore(3)  # Process up to 3 resumes concurrently
#
#         async def process_with_semaphore(resume):
#             async with semaphore:
#                 try:
#                     return await process_resume_url(resume, job_description, api_key, session)
#                 except Exception as e:
#                     logger.error(f"Error in process_with_semaphore for user {resume.user_id}: {str(e)}")
#                     return {
#                         "user_id": resume.user_id,
#                         "score": 0.0,
#                         "sgAnalysis": "Could not analyze skills gap due to processing error."
#                     }
#
#         # Create tasks for all resumes
#         tasks = [process_with_semaphore(resume) for resume in resumes]
#
#         # Execute all tasks
#         results = await asyncio.gather(*tasks)
#
#         # Sort results by score
#         ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
#
#         return ranked_results
#
#
# # Endpoints
# @app.post("/analyze", response_model=ATSResponse,
#           summary="Analyze resumes against a job description",
#           description="Provide resume URLs with user IDs and a job description to get ATS analysis")
# async def analyze_resumes_endpoint(
#         request: ATSRequest,
#         api_key: str = Depends(get_api_key)
# ):
#     if not request.resumes:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="No resume URLs provided"
#         )
#
#     logger.info(f"Received analyze request with {len(request.resumes)} resumes")
#     results = await process_resumes(request.resumes, request.job_description, api_key)
#     logger.info(f"Analysis complete for {len(results)} resumes")
#     return {"results": results}
#
#
# # Form-based endpoint for backward compatibility
# @app.post("/analyze-form", response_model=ATSResponse,
#           summary="Analyze resumes against a job description (form-based)",
#           description="Provide resume URLs with user IDs and a job description using form data")
# async def analyze_resumes_form(
#         job_description: str = Form(...),
#         resume_urls: str = Form(...),  # Comma-separated URLs
#         user_ids: str = Form(...),  # Comma-separated user IDs
#         api_key: str = Depends(get_api_key)
# ):
#     # Parse resume URLs and user IDs from form data
#     urls = [url.strip() for url in resume_urls.split(",") if url.strip()]
#     ids = [uid.strip() for uid in user_ids.split(",") if uid.strip()]
#
#     if not urls:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="No resume URLs provided"
#         )
#
#     if len(urls) != len(ids):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="The number of resume URLs must match the number of user IDs"
#         )
#
#     logger.info(f"Received form-based analyze request with {len(urls)} resumes")
#
#     # Convert to ResumeInput objects with user IDs
#     resume_inputs = [ResumeInput(url=url, user_id=uid) for url, uid in zip(urls, ids)]
#
#     results = await process_resumes(resume_inputs, job_description, api_key)
#     logger.info(f"Form-based analysis complete for {len(results)} resumes")
#     return {"results": results}
#
#
# # Health check endpoint
# @app.get("/health",
#          summary="Health check endpoint",
#          description="Check if the API is running")
# def health_check():
#     return {"status": "healthy"}
#
#
# # Main driver
# if __name__ == "__main__":
#     import uvicorn
#
#     logger.info("Starting ATS Resume Analyzer API server")
#     uvicorn.run(app, host="0.0.0.0", port=8000)
