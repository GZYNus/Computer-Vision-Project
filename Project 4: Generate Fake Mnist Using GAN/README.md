# Generate Fake Mnist Using GAN (Generative Adversarial Network)

* Main Idea: 
In GAN, we need to build two models separately, Generator and Discriminator. Also, we need to train these 2 models separately. A generator is to create fake mnist figures, originated from random noise. We give the label for these fake figures all positive(1), which means the generator will try its best to fit the ground truth mnist figures, which are obviously labeled positive(1). When we are training Generator, the discriminator should not do parameter optimization. Besides, as to Discriminator, we will label fake mnist as 0 and ground truth mnist as 0.9(for better performance). Thus, the discriminator will also try its best to find out whether they are fake or not. Hence, Generator and Discriminator form a relationship of adversary.

![](https://github.com/GZYNus/Computer-Vision-Project/blob/master/Project%204:%20Generate%20Fake%20Mnist%20Using%20GAN/GAN.jpg)

* GAN generated numerical images after 1 Epoch: (100 images in total)

![](https://github.com/GZYNus/Computer-Vision-Project/blob/master/Project%204:%20Generate%20Fake%20Mnist%20Using%20GAN/gan_generated_image_epoch_1.png)

* GAN generated numerical images after 20 Epoch: (100 images in total)

![](https://github.com/GZYNus/Computer-Vision-Project/blob/master/Project%204:%20Generate%20Fake%20Mnist%20Using%20GAN/gan_generated_image_epoch_20.png)

* GAN generated numerical images after 40 Epoch: (100 images in total)

![](https://github.com/GZYNus/Computer-Vision-Project/blob/master/Project%204:%20Generate%20Fake%20Mnist%20Using%20GAN/gan_generated_image_epoch_40.png)

* Note that here I use mini-batch gradient descent(Adam). Hence during 1 epoch, the network will update 234 times (60000/256).
