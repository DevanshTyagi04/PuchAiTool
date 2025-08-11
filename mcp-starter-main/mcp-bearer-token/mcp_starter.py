import asyncio
import os
from typing import Annotated
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl
import markdownify
import httpx
import readabilipy
import openai

# --- Load environment variables ---
load_dotenv()
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert OPENAI_API_KEY is not None, "Please set OPENAI_API_KEY in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break
        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )
    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )
    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )
    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))

# --- Image: Black and White ---
MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64, io
    from PIL import Image
    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))
        bw_image = image.convert("L")
        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")
        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Tool: Summarize File (with OpenAI API) ---
@mcp.tool(description="Summarize uploaded files (PDF, Word, TXT) and extract key points using AI.")
async def summarize_file(
    file_data: Annotated[str, Field(description="Base64-encoded file data")],
    file_type: Annotated[str, Field(description="File type: pdf, docx, or txt")],
) -> str:
    import base64, io
    try:
        content = ""
        file_bytes = base64.b64decode(file_data)
        buf = io.BytesIO(file_bytes)
        if file_type == "pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(buf)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif file_type == "docx":
            from docx import Document
            doc = Document(buf)
            content = "\n".join(para.text for para in doc.paragraphs)
        elif file_type == "txt":
            content = buf.read().decode("utf-8")
        else:
            raise ValueError("Unsupported file type.")

        # OpenAI Text Summarization
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text concisely:\n{content[:6000]}"}
            ],
            max_tokens=200,
            temperature=0.3,
        )
        summary = response.choices[0].message["content"]
        return f"**Summary:**\n{summary}"
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Tool: Image Generator/Editor (with OpenAI DALL·E) ---
@mcp.tool(description="Generate an image from a prompt, or edit an uploaded image (resize, filter).")
async def image_tool(
    prompt: Annotated[str | None, Field(description="Image generation prompt")]=None,
    uploaded_image: Annotated[str | None, Field(description="Base64-encoded image to edit")]=None,
    resize_width: Annotated[int | None, Field(description="Optional: resize image to this width")]=None,
    resize_height: Annotated[int | None, Field(description="Optional: resize image to this height")]=None,
    grayscale: Annotated[bool, Field(description="Optional: apply grayscale filter")]=False,
) -> list[ImageContent]:
    import base64, io
    from PIL import Image
    if prompt is not None:
        # OpenAI DALL·E Image Generation
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512",
            response_format="b64_json",
        )
        generated_base64 = response["data"][0]["b64_json"]
        return [ImageContent(type="image", mimeType="image/png", data=generated_base64)]
    if uploaded_image is not None:
        image_bytes = base64.b64decode(uploaded_image)
        image = Image.open(io.BytesIO(image_bytes))
        if resize_width and resize_height:
            image = image.resize((resize_width, resize_height))
        if grayscale:
            image = image.convert("L")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        out_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return [ImageContent(type="image", mimeType="image/png", data=out_base64)]
    raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide a prompt or an uploaded image to process."))

# --- Tool: File Converter (PDF <-> DOCX) ---
@mcp.tool(description="Convert between PDF and DOCX file formats (upload base64 file, get converted base64).")
async def convert_file(
    input_file: Annotated[str, Field(description="Base64-encoded source file")],
    input_type: Annotated[str, Field(description="Source file type: pdf or docx")],
    output_type: Annotated[str, Field(description="Output file type: pdf or docx")],
) -> TextContent:
    import base64, io, os, tempfile
    try:
        inp_bytes = base64.b64decode(input_file)
        with tempfile.TemporaryDirectory() as tmpdir:
            inp_path = os.path.join(tmpdir, f"input.{input_type}")
            out_path = os.path.join(tmpdir, f"output.{output_type}")
            with open(inp_path, "wb") as f:
                f.write(inp_bytes)
            # DOCX → PDF
            if input_type == "docx" and output_type == "pdf":
                from docx2pdf import convert
                convert(inp_path, out_path)
            # PDF → DOCX
            elif input_type == "pdf" and output_type == "docx":
                from pdf2docx import Converter
                cv = Converter(inp_path)
                cv.convert(out_path)
                cv.close()
            else:
                raise ValueError("Unsupported conversion.")
            with open(out_path, "rb") as f:
                out_b64 = base64.b64encode(f.read()).decode("utf-8")
        return TextContent(type="text", text=out_b64)
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
