
# Inspiration

CORVID-19 has changed our lifestyle. The safest and easiest defence against it is maintaining the social distancing. CDC guideline says to maintain a 6ft distance in public places.

Groceries, Pharmacies and Restaurants now have 6ft lines on the ground. But can we keep drawing lines all over the floor ? Is there a more effective, cleaner and elegant solution for this?

Inspired by how speed limit is reminded using display boards. This method gently reminds people of their responsibility to public safety. We can use the existing CCTV infrastructure in stores and use a big monitor to display back whether or not 6 ft distance rule is followed

# What it does

The app uses the existing CCTV infrastructure in stores and use a big monitor to display back whether or not 6 ft distance rule is followed

# How I built it

The app takes a CCTV video as the input and run each frame through YOLOv3 deep learning network to detect people and draw a bounding box.

The frame is then mapped to a bird eye view to calculate the distance from the person to the camera

These distance values are used to calculate distance between two people using the cosine rule

# Challenges I ran into

OMG! Time is not enough

# Accomplishments that I'm proud of

Building something to help the society using the advanced technologies like machine learing
