from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.llm import get_llm



EMAIL_PROMPT = PromptTemplate(
    input_variables=["email_type", "context", "tone", "recipient", "sender"],
    template="""You are an expert professional email writer.

    Write a {email_type} email with the following details:
    - Recipient: {recipient}
    - Sender: {sender}
    - Tone: {tone}
    - Context/Purpose: {context}
    Requirements:
    - Write a clear subject line
    - Keep it professional and concise
    - Match the requested tone exactly
    - Include proper greeting and sign-off

Email:"""
)

COVER_LETTER_PROMPT = PromptTemplate(
    input_variables=["job_title", "company", "skills", "experience", "name"],
    template="""You are an expert cover letter writer.

Write a compelling cover letter for the following:
- Applicant Name: {name}
- Job Title: {job_title}
- Company: {company}
- Key Skills: {skills}
- Experience Summary: {experience}

Requirements:
- Start with a strong opening paragraph
- Highlight relevant skills and experience
- Show enthusiasm for the role and company
- End with a clear call to action
- Keep it to 3-4 paragraphs

Cover Letter:"""
)


def write_email(
    email_type: str,
    context: str,
    tone: str = "professional",
    recipient: str = "Team",
    sender: str = "Your Name"
) -> str:
    """
    Generates a professional email.

    Args:
        email_type: Type of email (e.g. follow-up, apology, introduction)
        context: Purpose or context of the email
        tone: Tone of the email (professional, friendly, formal)
        recipient: Name of the recipient
        sender: Name of the sender

    Returns:
        Generated email string
    """
    if not context.strip():
        return "⚠️ Please provide context for the email."
    llm = get_llm()
    chain = EMAIL_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "email_type": email_type,
        "context": context,
        "tone": tone,
        "recipient": recipient,
        "sender": sender
    })

def write_cover_letter(
    job_title: str,
    company: str,
    skills: str,
    experience: str,
    name: str = "Your Name"
) -> str:
    """
    Generates a cover letter.

    Args:
        job_title: Job title applying for
        company: Company name
        skills: Key skills relevant to the role
        experience: Brief experience summary
        name: Applicant name

    Returns:
        Generated cover letter string
    """
    if not job_title.strip() or not company.strip():
        return "⚠️ Please provide job title and company name."

    llm = get_llm()
    chain = COVER_LETTER_PROMPT | llm | StrOutputParser()

    return chain.invoke({
        "job_title": job_title,
        "company": company,
        "skills": skills,
        "experience": experience,
        "name": name
    })
    