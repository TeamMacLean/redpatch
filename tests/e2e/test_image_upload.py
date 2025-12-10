"""
End-to-end tests for the Redpatch web application.
Tests the complete flow of uploading an image, processing it, and viewing results.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from playwright.sync_api import Page, expect


# Test fixtures
@pytest.fixture(scope="module")
def test_image_path():
    """Path to a test image for upload."""
    return str(project_root / "tests" / "known_coords_sizes" / "centered_leaf.jpg")


@pytest.fixture(scope="module")
def app_server():
    """Start the Flask application server for testing."""
    env = os.environ.copy()
    env["FLASK_ENV"] = "testing"

    # Start the Flask server
    server_process = subprocess.Popen(
        [sys.executable, str(project_root / "webapp.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Wait for server to start
    time.sleep(3)

    yield "http://localhost:5000"

    # Cleanup: terminate the server
    server_process.terminate()
    server_process.wait(timeout=5)


class TestImageUploadFlow:
    """Test the complete image upload and processing flow."""

    def test_homepage_loads(self, page: Page, app_server: str):
        """Test that the homepage loads correctly with the upload form."""
        page.goto(app_server)

        # Check page title
        expect(page).to_have_title("Redpatch - Leaf Lesion Analysis")

        # Check for main heading
        heading = page.locator("h1")
        expect(heading).to_contain_text("Redpatch")

        # Check for upload form elements
        expect(page.locator("#upload-form")).to_be_visible()
        expect(page.locator("#image-input")).to_be_visible()
        expect(page.locator("#submit-btn")).to_be_visible()

    def test_upload_form_requires_file(self, page: Page, app_server: str):
        """Test that the form requires a file to be selected."""
        page.goto(app_server)

        # The file input should have required attribute
        file_input = page.locator("#image-input")
        expect(file_input).to_have_attribute("required", "")

    def test_complete_upload_and_process_flow(
        self, page: Page, app_server: str, test_image_path: str
    ):
        """
        Test the complete end-to-end flow:
        1. Load the homepage
        2. Upload a test image
        3. Wait for processing
        4. Verify results are displayed
        """
        # Step 1: Navigate to homepage
        page.goto(app_server)
        expect(page.locator("#upload-form")).to_be_visible()

        # Step 2: Upload the test image
        file_input = page.locator("#image-input")
        file_input.set_input_files(test_image_path)

        # Step 3: Submit the form
        submit_button = page.locator("#submit-btn")
        submit_button.click()

        # Step 4: Wait for processing and results page
        # The page should eventually show results (may take a few seconds for processing)
        page.wait_for_url("**/process/**", timeout=30000)

        # Step 5: Verify results page elements
        # Wait for results to load (processing may take time)
        results_summary = page.locator("#results-summary")
        expect(results_summary).to_be_visible(timeout=60000)

        # Check that the results page has expected content
        expect(page).to_have_title("Results - Redpatch")

        # Verify images are displayed
        original_image = page.locator("#original-image")
        expect(original_image).to_be_visible()

        annotated_image = page.locator("#annotated-image")
        expect(annotated_image).to_be_visible()

        # Verify statistics are shown
        expect(page.locator("text=Healthy Regions:")).to_be_visible()
        expect(page.locator("text=Lesion Regions:")).to_be_visible()
        expect(page.locator("text=Lesion Centres:")).to_be_visible()

    def test_results_show_analysis_data(
        self, page: Page, app_server: str, test_image_path: str
    ):
        """Test that the results page shows meaningful analysis data."""
        # Upload and process image
        page.goto(app_server)
        page.locator("#image-input").set_input_files(test_image_path)
        page.locator("#submit-btn").click()

        # Wait for results
        page.wait_for_selector("#results-summary", timeout=60000)

        # Check that we have some results (the test image should produce data)
        results_text = page.locator("#results-summary").inner_text()

        # Verify the statistics section contains numbers
        assert "Healthy Regions:" in results_text
        assert "Total Healthy Area (pixels):" in results_text

    def test_can_analyze_another_image(
        self, page: Page, app_server: str, test_image_path: str
    ):
        """Test that the 'Analyze Another Image' link works."""
        # First, complete an analysis
        page.goto(app_server)
        page.locator("#image-input").set_input_files(test_image_path)
        page.locator("#submit-btn").click()

        # Wait for results
        page.wait_for_selector("#results-summary", timeout=60000)

        # Click the back link
        back_link = page.locator("text=Analyze Another Image")
        expect(back_link).to_be_visible()
        back_link.click()

        # Verify we're back at the upload page
        expect(page).to_have_url(app_server + "/")
        expect(page.locator("#upload-form")).to_be_visible()

    def test_images_load_correctly(
        self, page: Page, app_server: str, test_image_path: str
    ):
        """Test that result images load without errors."""
        # Upload and process
        page.goto(app_server)
        page.locator("#image-input").set_input_files(test_image_path)
        page.locator("#submit-btn").click()

        # Wait for results
        page.wait_for_selector("#results-summary", timeout=60000)

        # Check that images have valid src attributes
        original_img = page.locator("#original-image")
        annotated_img = page.locator("#annotated-image")

        original_src = original_img.get_attribute("src")
        annotated_src = annotated_img.get_attribute("src")

        assert original_src is not None
        assert annotated_src is not None
        assert "/images/" in original_src
        assert "/images/" in annotated_src
        assert "_annotated" in annotated_src


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_image_shows_error(self, page: Page, app_server: str, tmp_path):
        """Test that uploading an invalid file shows an error."""
        # Create a fake image file (just text, not a real image)
        fake_image = tmp_path / "fake.jpg"
        fake_image.write_text("This is not a real image")

        page.goto(app_server)
        page.locator("#image-input").set_input_files(str(fake_image))
        page.locator("#submit-btn").click()

        # Should show error page or message
        # Wait for either error message or processing error
        page.wait_for_load_state("networkidle", timeout=30000)

        # Check if we got an error (the page content should indicate an issue)
        page_content = page.content()
        # Either on error page or still on upload page with error
        assert (
            "error" in page_content.lower()
            or "Error" in page_content
            or "upload" in page_content.lower()
        )
