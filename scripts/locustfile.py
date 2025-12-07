from locust import HttpUser, task, between

class FastAPITestUser(HttpUser):
    wait_time = between(1, 2)  # Wait 1-2 seconds between tasks

    @task
    def predict_image(self):
        with open("C:/Users/Dell/Documents/Freelancing template/Image_classification_template2/data/sample_image5.jpg", "rb") as image_file:


            self.client.post(
                "/predict-image/",
                files={"file": ("C:/Users/Dell/Documents/Freelancing template/Image_classification_template2/data/sample_image5.jpg", image_file, "image/jpeg")}
            )
