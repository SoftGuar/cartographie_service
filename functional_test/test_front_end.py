from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import pytest

class TestIndoorMapping:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Set up Chrome options
        chrome_options = Options()
        # chrome_options.add_argument("--headless")  # Commented out for debugging
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")

        # Initialize the Chrome WebDriver with the configured options
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        self.wait = WebDriverWait(self.driver, 20)
        self.driver.get("http://localhost:5173")

        yield

        if self.driver:
            self.driver.quit()

    def test_create_and_process_room(self):
        """
        Test the complete flow of creating a room and processing its floor plan
        """
        try:
            # Wait for map to load completely
            time.sleep(2)
            
            # 1. Click on the map to create a new room
            map_element = self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "leaflet-container"))
            )
            
            # Get map dimensions
            map_size = map_element.size
            
            # Calculate center coordinates
            center_x = map_size['width'] // 2
            center_y = map_size['height'] // 2
            
            # Click near the center of the map
            actions = ActionChains(self.driver)
            actions.move_to_element(map_element)
            actions.move_by_offset(10, 10)  # Move to exact center
            actions.click()
            actions.perform()

            # 2. Fill and submit the room form
            self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//h2[text()='Add New Room']"))
            )
            
            # Fill form fields
            name_input = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@placeholder='e.g., Lab 1']"))
            )
            name_input.send_keys("Test Room")
            
            # building_input = self.driver.find_element(By.XPATH, "//input[@placeholder='e.g., Research Center']")
            # building_input.send_keys("Test Building")
            
            # floor_input = self.driver.find_element(By.XPATH, "//label[text()='Floor']/following-sibling::input")
            # floor_input.clear()
            # floor_input.send_keys("1")
            
            # Submit form
            submit_button = self.driver.find_element(By.XPATH, "//button[text()='Create Room']")
            self.driver.execute_script("arguments[0].click();", submit_button)

            # 3. Wait for room details modal and upload floor plan
            self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//h2[text()='Test Room']"))
            )
            
            # Upload floor plan image
            current_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(current_dir, "test3.png")
            
            # Ensure the file exists
            if not os.path.exists(image_path):
                # Create a simple test image if it doesn't exist
                from PIL import Image
                img = Image.new('RGB', (100, 100), color='white')
                img.save(image_path)
            
            file_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
            file_input.send_keys(image_path)

            # # 4. Wait for image processing and rotate image
            # self.wait.until(
            #     EC.invisibility_of_element_located((By.CLASS_NAME, "animate-spin"))
            # )
            
            # Find rotation buttons
            rotation_buttons = self.driver.find_elements(
                By.XPATH, 
                "//button[contains(@class, 'rotation')]"
            )
            
            # Rotate clockwise
            self.driver.execute_script("arguments[0].click();", rotation_buttons[1])
            time.sleep(0.5)
            
            # Rotate counter-clockwise
            self.driver.execute_script("arguments[0].click();", rotation_buttons[0])
            time.sleep(0.5)
            
            # Click validate rotation button
            validate_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'bg-green-500')]"))
            )
            self.driver.execute_script("arguments[0].click();", validate_button)

            # 5. Wait for processing to complete
            self.wait.until(
                EC.invisibility_of_element_located((By.CLASS_NAME, "animate-spin"))
            )

            # 6. Modify floor plan
            # Click erase button
            # erase_button = self.wait.until(
            #     EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Erase')]"))
            # )
            # self.driver.execute_script("arguments[0].click();", erase_button)
            
            # Find canvas and perform erase action
            canvas = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//canvas[contains(@class, 'absolute')]"))
            )
            
            # Get canvas dimensions
            canvas_size = canvas.size
            
            # # Calculate center coordinates
            canvas_center_x = canvas_size['width'] // 2
            canvas_center_y = canvas_size['height'] // 2
            
            # # Perform erase action near the center
            actions = ActionChains(self.driver)
            actions.move_to_element(canvas)
            actions.move_by_offset(10, 10)
            actions.click_and_hold()
            actions.move_by_offset(20, 20)
            actions.release()
            actions.perform()
            time.sleep(0.5)

            # Click add obstacle button
            time.sleep(1)
            add_obstacle = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[span[contains(text(), 'Add Obstacle')]]"))
            )
            self.driver.execute_script("arguments[0].click();", add_obstacle)
            
            # Draw obstacle
            actions = ActionChains(self.driver)
            actions.move_to_element_with_offset(canvas, 1, 1)
            actions.click_and_hold()
            actions.move_by_offset(2, 2)
            actions.release()
            actions.perform()
            time.sleep(1)

            # 7. Save the floor plan
            save_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Save Map')]"))
            )
            self.driver.execute_script("arguments[0].click();", save_button)

            # Wait for success message
            self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Map saved successfully')]"))
            )

        except Exception as e:
            # Take screenshot on failure
            self.driver.save_screenshot("test_failure.png")
            raise e