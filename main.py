import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
from matplotlib import pyplot as plt

from torch import optim, nn


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel(784, 10)
    trainloader, testloader = mnist(shuffle=False)

    n_epochs = 40

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.3)

    steps = 0
    losses = []
    for e in range(n_epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        else:
            losses.append(running_loss/len(trainloader))
            steps += 1
            print(f"Training loss: {running_loss/len(trainloader)}")

    steps = [i for i in range(steps)]

    # Use the plot function to draw a line plot
    plt.plot(steps, losses)

    # Add a title and axis labels
    plt.title("Training Loss vs Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss")

    # Show the plot
    plt.show()

    # Test data is loading good enough
    # i = 0
    # for images, labels in trainloader:
    #     i+=1

    #     print(labels[0])

    #     if i >= 5: break
    #     images = images.reshape(64, 28,28)
    #     plt.imshow(images[0], interpolation='nearest')
    #     plt.show()

    torch.save(model.state_dict(), 'trained_model.pt')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel(784, 10)
    model.load_state_dict(torch.load(model_checkpoint))
    _, testloader = mnist()

    # print(model)


    with torch.no_grad():
        for images, labels in testloader:
            # resize images
            images = images.view(images.shape[0], -1)

            model.eval()
            logps = model.forward(images)
            ps = torch.exp(logps)

            # Take max from the probs
            top_p, top_class = ps.topk(1, dim=1)

            # Compare with labels
            equals = top_class == labels.view(*top_class.shape)

            # mean
            accuracy = torch.mean(equals.type(torch.FloatTensor))

    print(f'Accuracy: {accuracy.item()*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
