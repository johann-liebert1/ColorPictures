import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
from io import BytesIO

# Define paths
prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = 'models/pts_in_hull.npy'

# Load the network
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load cluster centers
points = np.load(kernel_path)
points = points.transpose().reshape(2, 313, 1, 1)

# Update layer blobs
try:
    class8_ab_layer = net.getLayer(net.getLayerId("class8_ab"))
    class8_ab_layer.blobs = [points.astype(np.float32)]

    conv8_313_rh_layer = net.getLayer(net.getLayerId("conv8_313_rh"))
    conv8_313_rh_layer.blobs = [np.full([1, 313], 2.606, dtype="float32")]
except Exception as e:
    print(f"Error updating layer blobs: {e}")
    raise

# Define a function to colorize an image
def colorize_image(bwImg):
    # Preprocess the image
    normalized = bwImg.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resize = cv2.resize(lab, (224, 224))
    L = cv2.split(resize)[0]
    L -= 50

    # Set the input to the network
    net.setInput(cv2.dnn.blobFromImage(L))

    # Perform forward pass
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the output
    ab = cv2.resize(ab, (bwImg.shape[1], bwImg.shape[0]))

    # Concatenate with L channel and convert back to BGR
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")

    # Convert to bytes
    is_success, buffer = cv2.imencode(".jpg", colorized)
    io_buf = BytesIO(buffer)

    return colorized, io_buf

# Define a function to display the input and output images
def display_images(bwImg, colorizedImg):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(cv2.cvtColor(bwImg, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Grayscale Image")
    axs[1].imshow(cv2.cvtColor(colorizedImg, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Colorized Image")
    plt.show()

# Define a Streamlit app
def app():
    st.title("Image Colorization App")
    st.header("Upload Black and White Images to make them coloured!")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        colorizedImg, io_buf = colorize_image(image)
        st.write("### Input Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        st.write("### Colorized Image")
        st.image(cv2.cvtColor(colorizedImg, cv2.COLOR_BGR2RGB))

        # Add download button
        st.download_button(
            label="Download Colorized Image",
            data=io_buf,
            file_name="colorized_image.jpg",
            mime="image/jpeg"
        )

app()





# CODE -1


# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# import cv2
# # Define paths
# prototxt_path = "models/colorization_deploy_v2.prototxt"
# model_path = "models/colorization_release_v2.caffemodel"
# kernel_path = 'models/pts_in_hull.npy'

# # Load the network
# net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# # Load cluster centers
# points = np.load(kernel_path)
# points = points.transpose().reshape(2, 313, 1, 1)

# # Update layer blobs
# try:
#     class8_ab_layer = net.getLayer(net.getLayerId("class8_ab"))
#     class8_ab_layer.blobs = [points.astype(np.float32)]

#     conv8_313_rh_layer = net.getLayer(net.getLayerId("conv8_313_rh"))
#     conv8_313_rh_layer.blobs = [np.full([1, 313], 2.606, dtype="float32")]
# except Exception as e:
#     print(f"Error updating layer blobs: {e}")
#     raise

# # Define a function to colorize an image
# def colorize_image(bwImg):
#     # Preprocess the image
#     normalized = bwImg.astype("float32") / 255.0
#     lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
#     resize = cv2.resize(lab, (224, 224))
#     L = cv2.split(resize)[0]
#     L -= 50

#     # Set the input to the network
#     net.setInput(cv2.dnn.blobFromImage(L))

#     # Perform forward pass
#     ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

#     # Resize the output
#     ab = cv2.resize(ab, (bwImg.shape[1], bwImg.shape[0]))

#     # Concatenate with L channel and convert back to BGR
#     L = cv2.split(lab)[0]
#     colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
#     colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
#     colorized = (255.0 * colorized).astype("uint8")

#     return colorized

# # Define a function to display the input and output images
# def display_images(bwImg, colorizedImg):
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#     axs[0].imshow(cv2.cvtColor(bwImg, cv2.COLOR_BGR2RGB))
#     axs[0].set_title("Grayscale Image")
#     axs[1].imshow(cv2.cvtColor(colorizedImg, cv2.COLOR_BGR2RGB))
#     axs[1].set_title("Colorized Image")
#     plt.show()

# # Define a Streamlit app
# def app():
#     st.title("Image Colorization App")

#     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#     if uploaded_image is not None:
#         image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
#         colorizedImg = colorize_image(image)
#         st.write("### Input Image")
#         st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         st.write("### Colorized Image")
#         st.image(cv2.cvtColor(colorizedImg, cv2.COLOR_BGR2RGB))
      
#         )
# app()

# def app():
#     # Load the input image
#     image_path = st.text_input("Enter the path to the input image:")
#     if image_path:
#         bwImg = cv2.imread(image_path)
#         if bwImg is None:
#             st.write(f"Error: Image file {image_path} not found.")
#         else:
#             # Colorize the image
#             colorizedImg = colorize_image(bwImg)

#             # Display the input and output images
#             st.write("### Input Image")
#             st.image(cv2.cvtColor(bwImg, cv2.COLOR_BGR2RGB))
#             st.write("### Colorized Image")
#             st.image(cv2.cvtColor(colorizedImg, cv2.COLOR_BGR2















# # CODE 1:

# import numpy as np
# import cv2
# import streamlit as st

# # Define paths
# prototxt_path = "models/colorization_deploy_v2.prototxt"
# model_path = "models/colorization_release_v2.caffemodel"
# kernel_path = 'models/pts_in_hull.npy'

# # Load the network
# net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# # Load cluster centers
# points = np.load(kernel_path)
# points = points.transpose().reshape(2, 313, 1, 1)

# # Function to colorize the image
# def colorize_image(image):
#     # Preprocess the image
#     normalized = image.astype("float32") / 255.0
#     lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
#     resize = cv2.resize(lab, (224, 224))
#     L = cv2.split(resize)[0]
#     L -= 50

#     # Set the input to the network
#     net.setInput(cv2.dnn.blobFromImage(L))

#     # Perform forward pass
#     ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

#     # Resize the output
#     ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

#     # Concatenate with L channel and convert back to BGR
#     L = cv2.split(lab)[0]
#     colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
#     colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
#     colorized = (255.0 * colorized).astype("uint8")

#     return colorized

# # Streamlit app
# st.title("Image Colorization App")

# # Upload image
# uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:
#     # Read the uploaded image
#     image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

#     # Colorize the image
#     colorized_image = colorize_image(image)

#     # Display original and colorized images
#     st.image([image, colorized_image], caption=['Original Image', 'Colorized Image'], width=300)




# # CODE 2 : THIS CODE DOES NOT INCLUDE DEPLOYMENT !!


# # import numpy as np
# # import cv2
# # import matplotlib.pyplot as plt
# # import streamlit as st

# # # Define paths
# # prototxt_path = "models/colorization_deploy_v2.prototxt"
# # model_path = "models/colorization_release_v2.caffemodel"
# # kernel_path = 'models/pts_in_hull.npy'
# # image_path = 'pic3.jpg'

# # # Load the network
# # net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# # # Load cluster centers
# # points = np.load(kernel_path)
# # points = points.transpose().reshape(2, 313, 1, 1)

# # # Update layer blobs
# # try:
# #     class8_ab_layer = net.getLayer(net.getLayerId("class8_ab"))
# #     class8_ab_layer.blobs = [points.astype(np.float32)]

# #     conv8_313_rh_layer = net.getLayer(net.getLayerId("conv8_313_rh"))
# #     conv8_313_rh_layer.blobs = [np.full([1, 313], 2.606, dtype="float32")]
# # except Exception as e:
# #     print(f"Error updating layer blobs: {e}")
# #     raise

# # # Load the input image
# # bwImg = cv2.imread(image_path)
# # if bwImg is None:
# #     raise FileNotFoundError(f"Image file {image_path} not found.")

# # # Preprocess the image
# # normalized = bwImg.astype("float32") / 255.0
# # lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
# # resize = cv2.resize(lab, (224, 224))
# # L = cv2.split(resize)[0]
# # L -= 50

# # # Set the input to the network
# # net.setInput(cv2.dnn.blobFromImage(L))

# # # Perform forward pass
# # ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# # # Resize the output
# # ab = cv2.resize(ab, (bwImg.shape[1], bwImg.shape[0]))

# # # Concatenate with L channel and convert back to BGR
# # L = cv2.split(lab)[0]
# # colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
# # colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
# # colorized = (255.0 * colorized).astype("uint8")




