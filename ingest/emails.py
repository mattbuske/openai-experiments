import chunk
import email
import quopri

# Splits the extracted text from a email file into chunks
def chunk_email(file_path):

    # Open the Email File
    with open(file_path) as email_file:
        # Get the Email Message
        email_message = email.message_from_file(email_file)
        # Process The Email for Chunking
        extracted_text = "File Type: Email "
        # Get the Header Information
        extracted_text += "Header Information:"
        header_values = email_message.items()
        for header, value in header_values:
            extracted_text += f" {header}: {value}"
        # Get the email content
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == 'text/plain':
                    charset = part.get_content_charset()
                    if part.get('Content-Transfer-Encoding') == 'quoted-printable':
                        encoded_text = part.get_payload(decode=True)
                        body = quopri.decodestring(encoded_text).decode(charset)
                    else:
                        body = part.get_payload(decode=True).decode(charset)
                    extracted_text += f" Body: {body}"
                    break

    return chunk.get_chunks(extracted_text)