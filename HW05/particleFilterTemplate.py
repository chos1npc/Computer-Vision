import cv2
from PIL import Image, ImageDraw
import numpy as np
import math
 
def gaussian1D(m, s):
  r = np.random.normal(m,s,1) #s: standard deviation
  return r

def gaussian2D(m, s):
  r1 = np.random.normal(m,s,1) #s: standard deviation
  r2 = np.random.normal(m,s,1) #s: standard deviation
  return np.array([r1[0],r2[0]])

########   draw and show particles  #######
### numParticles : number of particles
### particles: particles state vectors
### frame: the image
### color: the particles color you assign

def display(numParticles, particles, frame, color): 
  for i in range(numParticles):
    # Get the position of the particle
    pos = particles[:2, i].astype(int)

    # Ensure the particle's position is within the frame's boundaries
    pos = np.clip(pos, 0, np.array([imgW-1, imgH-1]))

    # Draw the particle on the frame
    cv2.circle(frame, tuple(pos), 1, color, -1)

  # Convert the image from BGR to RGB
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Convert the OpenCV image to a PIL image
  img_pil = Image.fromarray(frame_rgb)

  # Show the image
  img_pil.show()

def display(numParticles, particles, frame, color): 
  # Convert the image from BGR to RGB
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Convert the OpenCV image to a PIL image
  img_pil = Image.fromarray(frame_rgb)
  draw = ImageDraw.Draw(img_pil)

  for i in range(numParticles):
    # Get the position of the particle
    pos = particles[:2, i].astype(int)

    # Ensure the particle's position is within the frame's boundaries
    pos = np.clip(pos, 0, np.array([imgW-1, imgH-1]))

    # Draw the particle on the PIL image
    draw.ellipse([tuple(pos-1), tuple(pos+1)], fill=tuple(color))

  return np.array(img_pil.convert('RGB'))
cap = cv2.VideoCapture('CV_beginner\Person.wmv')

imgH = 480
imgW = 640

targetColor = np.array([0,0,255]) #red
colorSigma = 50 #TODO set the value ###set sigma of color likelihood function, try 50 first
posNoise = 15 #TODO  set the value ### set position uncertainty after prediction, try 15 first
velNoise = 5 #TODO  set the value ### set velocity uncertainty after prediction, try 5 first

numParticles = 1000  #you can chage it 

#particle state vectors and initialization
particles = np.array([np.random.random_integers(0,imgW-1, numParticles), np.random.random_integers(0,imgH-1, numParticles), 3 * np.random.randn(numParticles) + 3, 3 * np.random.randn(numParticles) ] )
#weights and initalization
weights = np.zeros(numParticles) + 1.0/numParticles
#prediction matrix (constant velocity model)
predMat = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (imgW, imgH))
# Read until video is completed
frameCount = 0
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  rawFrame = np.copy(frame)
  if ret == True:
    
    frameCount = frameCount + 1
    if frameCount < 30:  ### start the particle filter from the 30th frames, simplify the process
      continue

    ##### calculate likelihood of the particles by pixel color and update the particle's weight
    for i in range(numParticles):
      # Get the position of the particle
      pos = particles[:2, i].astype(int)

      # Ensure the particle's position is within the frame's boundaries
      pos = np.clip(pos, 0, np.array([imgW-1, imgH-1]))

      # Get the color of the particle's position in the frame
      particleColor = frame[pos[1], pos[0]]

      # Calculate the color difference
      colorDiff = np.linalg.norm(particleColor - targetColor)

      # Calculate the likelihood of the particle and update its weight
      weights[i] *= np.exp(-0.5 * (colorDiff ** 2) / (colorSigma ** 2))

    # Normalize the weights so they sum to 1
    weights /= weights.sum()


    ##### Resampleing: the particle with higher weight will be duplicated more times
    ##### One suggested steps: 
    #       1. normalization (weights): make sum of all weights of particles to 1
    #       2. create a temporary state vectors (particleTemp)
    #       3. a loop (k) (loop through all particle state vector in "particleTemp" one by one)
    #         3-1 randomize an "particleIndex" by following the distribution reprsented by "weights" (if the particle with a higher weight, its index will be returned with a higher chance) (check the function numpy.random.choice() )
    #         3-2 copy the "particleIndex"-th state vector in "particles" to the "k-th" state vector in "particleTemp"
    #       4. copy whole particleTemp back to particles
    #       This steps may be easier to implement. But it is well approximate to the resampling step we say in the lecture.
    #### TODO......
    # 1. Normalization of weights is already done in the previous step. If not, do it here:
    weights /= weights.sum()

    # 2. Create a temporary state vectors
    particleTemp = np.zeros_like(particles)

    # 3. Loop through all particle state vectors in particleTemp one by one
    for k in range(numParticles):
        # 3-1 Randomize a particleIndex following the distribution represented by weights
        particleIndex = np.random.choice(np.arange(numParticles), p=weights)

        # 3-2 Copy the particleIndex-th state vector in particles to the k-th state vector in particleTemp
        particleTemp[:, k] = particles[:, particleIndex]

    # 4. Copy whole particleTemp back to particles
    particles = particleTemp

    # Reset the weights to be equal for all particles
    weights.fill(1.0 / numParticles)


    ##reset weights
    weights = np.zeros(numParticles) + 1.0/numParticles
    
    frame_output = display(numParticles, particles, frame, (255,0,0)) #draw and show particles' location
    out.write(frame_output)
    ##### predict new position of a particle by its velocity and old position (plus noise) (constant velocity model)
    for i in range(numParticles):
      # Get the current position and velocity of the particle
      pos = particles[:2, i]
      vel = particles[2:, i]

      # Predict the new position by adding the velocity to the current position
      newPos = pos + vel

      # Add some noise to the prediction
      newPos += np.random.normal(0, posNoise, 2)

      # Update the position of the particle
      particles[:2, i] = newPos

      # Add some noise to the velocity
      particles[2:, i] += np.random.normal(0, velNoise, 2)


    # # Press Q on keyboard to  exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #   break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
out.release()
# # Closes all the frames
# cv2.destroyAllWindows()