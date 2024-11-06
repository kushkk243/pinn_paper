from robocorp.tasks import task
from robocorp import browser
from RPA.Assistant import Assistant
from model import *

@task
def spare_bin():
    model_train()
    

def model_train():
    assistant = Assistant()
    assistant.add_heading("Training Data Line")
    assistant.add_text_input("xstart", placeholder="Enter Starting X Position")
    assistant.add_text_input("xend", placeholder="Enter End X Position")
    assistant.add_text_input("ystart", placeholder="Enter Starting Y Position")
    assistant.add_text_input("yend", placeholder="Enter End Y Position")
    assistant.add_heading("Model Training")
    assistant.add_text_input("link1", placeholder="Enter length of link 1")
    assistant.add_text_input("link2", placeholder="Enter length of link 1")
    assistant.add_text_input("theta1", placeholder="Enter Angle Theta 1")
    assistant.add_text_input("theta2", placeholder="Enter Angle Theta 2")
    assistant.add_submit_buttons("Submit", default="Submit")
    result = assistant.run_dialog()
    print(result)
    model = CreateModel(float(result.xstart),float(result.ystart),float(result.xend),float(result.yend),l1=int(result.link1),l2=int(result.link2),theta1=int(result.theta1),theta2=int(result.theta2))
    losses = model.train()
    model.model_graphs(losses)
    assistant.add_heading('The Loss Plot of the training')
    assistant.add_image('loss.png')
    model.save_model()
    assistant.add_text("Your Model has been Trained and the weights have been saved as model.pt!",size='large')
    result = assistant.run_dialog()
    