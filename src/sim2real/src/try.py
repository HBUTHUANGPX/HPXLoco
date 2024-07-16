from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="H8WJeV3Ygo8migBZPL5Y"
)

result = CLIENT.infer("src/sim2real/src/WEBFire140_jpg.rf.6a2dc9f38cf34055a3a2519ba0da9f01.jpg", model_id="mytest-hrswj/1")