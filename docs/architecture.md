# Architecture

## Goal

This project is a self-contained local demo platform for practicing Hugging Face concepts.

Each video demo should include:

- a simple frontend interaction
- a local backend endpoint
- a Hugging Face concept
- sample inputs
- documented learning notes

## Runtime Flow

Browser frontend
→ FastAPI backend
→ Hugging Face model or utility
→ structured JSON response
→ browser result display

## Project Structure

backend/
- FastAPI app
- video-specific API routers
- reusable services
- backend tests

frontend/
- static HTML, CSS, and JavaScript
- dropdown navigation for selecting demos
- simple input and output panels

demos/
- video-specific documentation
- sample inputs
- notes about the Hugging Face concept

docs/
- architecture notes
- project planning
- playlist progress
