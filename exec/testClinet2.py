
import socket
import pickle
import math
import pygame as pg
class Client:
    def __init__(self, ipAddress, port, clientNumber):
        self.server = socket.socket()
        self.ipAddress = ipAddress
        self.port = port
        self.clientNumber = clientNumber
        try:
            self.server.connect((self.ipAddress, self.port))
            print("Connected to server")
        except Exception as e:
            print("something's wrong with %s:%d. Exception is %s" % (self.ipAddress, self.port, e))

    def send(self, data):
        self.server.send(pickle.dumps(data))

    def receive(self):
        data = self.server.recv(2048)
        allPositions = pickle.loads(data)
        return allPositions
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# This is a simple class that will help us print to the screen
# It has nothing to do with the joysticks, just outputting the
# information.
class TextPrint:
    def __init__(self):
        self.reset()
        self.font = pg.font.Font(None, 20)

    def print(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, [self.x, self.y])
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10

if __name__ == "__main__":


    import os
    import sys
    sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))

    ipAddress = '192.168.199.212'
    port = 12345
    clientNumber = int(input("Input your client number(0 or 1): "))
    client = Client(ipAddress, port, clientNumber)

    pg.init()



    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pg.time.Clock()

    # Initialize the joysticks
    pg.joystick.init()

    # Get ready to print
    textPrint = TextPrint()

    screenWidth = 800
    screenHeight = 800
    screenCenter = [screenWidth / 2, screenHeight / 2]
    fullScreen = False
    from src.visualization import DrawBackground, DrawNewState,InitializeScreen,DrawImage
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()
    gridSize = 60
    leaveEdgeSpace = 6
    lineWidth = 1
    from pygame.color import THECOLORS
    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [THECOLORS['blue']] * 16  # [255, 50, 50]
    playerColors = [THECOLORS['orange'], THECOLORS['red']]
    textColorTuple = THECOLORS['green']
    targetRadius = 10
    playerRadius = 10
    totalBarLength = 100
    barHeight = 20

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple, playerColors)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius)
    drawInstruction = DrawImage(screen)
    drawWaitPartner = DrawImage(screen,False,10)
    drawReadyImage = DrawImage(screen,False,1000)

    stopwatchEvent = pg.USEREVENT + 1
    stopwatchUnit = 100
    def receiveAction():
        reply = client.receive()
        if reply == "":
            return "null"
        else:
            return reply



    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'pictures'))
    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))

    waitPartnerImage = pg.image.load(os.path.join(picturePath, 'waitPartner.png'))
    waitPartnerImage = pg.transform.scale(waitPartnerImage, (screenWidth, screenHeight))

    readyImage = pg.image.load(os.path.join(picturePath, 'readyImage.png'))
    readyImage = pg.transform.scale(readyImage, (screenWidth, screenHeight))

    getReady = drawInstruction(introductionImage)

    waitReady = False
    while not waitReady:
        client.send((clientNumber, getReady))
        waitReady = receiveAction()
        # print(waitReady)
        drawWaitPartner(waitPartnerImage)


    drawReadyImage(readyImage)


    # -------- Main Program Loop -----------
    while done == False:
        # EVENT PROCESSING STEP
        for event in pg.event.get():  # User did something
            if event.type == pg.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

            # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
            elif event.type == stopwatchEvent:
                newStopwatch = newStopwatch + stopwatchUnit


        joystick_count = pg.joystick.get_count()



        # For each joystick:
        axesList = []
        for i in range(joystick_count):
            joystick = pg.joystick.Joystick(i)
            joystick.init()

            textPrint.print(screen, "Joystick {}".format(i))
            textPrint.indent()

            # Get the name from the OS for the controller/joystick

            # Usually axis run in pairs, up/down for one, and left/right for
            # the other.
            axes = joystick.get_numaxes()


            for i in range(axes):
                axis = joystick.get_axis(i)

                if abs(axis) > 0.5:
                    sign = axis/ abs(axis)
                    action = sign * math.log(9 * abs(axis) + 1) / 2.303
                else:
                    action =axis
                axesList.append(action)
        joystickSpaceSize = joystick_count * axes
        # print(axesList)
        axesList = [0 if abs(axesList[i]) < 0.5 else axesList[i] for i in range(len(axesList))]
        actionList = [axesList[i:i + 2] for i in range(0, len(axesList), axes)]
        # print(clientNumber)
        action=actionList[0]
        client.send((clientNumber,action))

        data = receiveAction()
        targetPositions=data[0]
        playerPositions=data[1]
        score=data[2]
        screen = drawNewState(targetPositions, playerPositions, 0, score)
        # Go ahead and update the screen with what we've drawn.
        pg.display.flip()

        # Limit to 20 frames per second
        clock.tick(20)

    # Close the window and quit.
    # If you forget this line, the program will 'hang'
    # on exit if running from IDLE.
    pg.quit()