FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
# This is required for Hugging Face Spaces
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Hugging Face Spaces require the app to listen on port 7860
EXPOSE 7860

# Run the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
