import google.genai

def test_generate_content():
    client = google.genai.GenerativeLanguageClient()

    # This is a placeholder example; replace with actual image bytes and model
    image_bytes = b"fake_image_bytes"

    contents = [
        google.genai.types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        ),
        "Caption this image."
    ]

    response = client.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    print("Response:", response)

if __name__ == "__main__":
    test_generate_content()
