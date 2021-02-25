__copyright__ = """
Machine Learning for Distributed Acoustic Sensing data (MLDAS)
Copyright (c) 2020, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
"""
__license__ = "Modified BSD license (see LICENSE.txt)"
__maintainer__ = "Vincent Dumont"
__email__ = "vincentdumont11@gmail.com"

import numpy
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from .prepare import set_creation

def decode_plot(model,datapath,img_size=100,vae=False,compare=False,adjust=False,discrepancy=0.1):
    """
    Show original and decoded images for 5 random images.
    """
    target = set_creation(datapath,img_size=img_size,nrand=5,adjust=adjust)
    model.eval()
    out, _ = model(target.float())
    plt.style.use('seaborn')
    plt.figure(figsize=(14,6*(1+int(compare))),dpi=80)
    for i in range(5):
        # Load and plot original image
        ax1 = plt.subplot(2*(1+int(compare)),5,i+1)
        ax1.imshow(target[i][0],cmap='viridis')
        ax1.set_title('Original image')
        normout = out[i][0]-out[i][0].min()
        normout = normout/normout.max()
        # Plot decoded image
        ax2 = plt.subplot(2*(1+int(compare)),5,6+i)
        ax2.imshow(normout.data,cmap='viridis')
        ax2.set_title('Decoded image'+(' (reference)' if i==0 and compare else ''))
        if compare:
            diff = normout-target[i][0]
            accuracy = len(diff[numpy.array(abs(diff)<discrepancy)])/args.img_size**2*100
            # Plot difference between decoded and original image
            ax3 = plt.subplot(4,5,11+i)
            ax3.imshow((normout-target[i][0]).data,cmap='seismic',vmin=-1,vmax=1)
            ax3.set_title('%.2f%% accuracy'%accuracy)
            # Plot difference between decoded and reference images
            if i==0:
                refout = normout
            ax4 = plt.subplot(4,5,16+i)
            ax4.imshow(abs(normout-refout).data,cmap='OrRd')
            ax4.set_title('Decoded - Reference')
    plt.tight_layout()
    plt.show()

def embedding_plot(model,datapath,img_size=100,stride=10,sample_size=1,nrand=None,adjust=True,show_images=True):
    """
    Display latent representation in 2D space.
    """
    data = set_creation(datapath,img_size,stride,sample_size,nrand=nrand,adjust=adjust)
    model.eval()
    z = model(data[:nrand].float())[-1]
    z = z.data.cpu().numpy()
    plt.style.use('seaborn')
    fig, ax = plt.subplots(dpi=100)
    plt.scatter(z[:,0,0], z[:,0,1])
    if show_images:
        for i in range(len(z)):
            imagebox = OffsetImage(data[i,0], zoom=0.4)
            ab = AnnotationBbox(imagebox, (z[i,0,0], z[i,0,1]),frameon=False)
            ax.add_artist(ab)
    plt.xlabel('Latent variable 1')
    plt.ylabel('Latent variable 2')
    plt.tight_layout()

def epoch_recon(models,datapath,img_size=100,adjust=False,epochs=120,max_diff=0.1,step_size=0.01):
    """
    Display image reconstruction accuracy across epochs.
    """
    target = set_creation(datapath,img_size=img_size,nrand=1000,adjust=adjust)                                    # Load 1000 images
    out,results = success_rate(model,target,img_size,args.discrepancy)
    acc_size = numpy.arange(0,max_diff,step_size)                                                                 # Define discrepancy ranges
    results = numpy.zeros((epochs,len(acc_size)))                                                                 # Initialize results array (epoch vs. reconstruction accuracy)
    for epoch in range(epochs):                                                                                   # Loop over epochs
        model_epoch = models[epoch+1]                                                                               # Load epoch model
        model_epoch.eval()                                                                                          # Set model to evaluation mode
        out, _ = model_epoch(target.float())                                                                        # Execute trained model to data
        for j in range(len(out)):                                                                                   # Loop over all output data
            out[j][0] = (out[j][0]-out[j][0].min())/(out[j][0].max()-out[j][0].min())                                 # Normalized outputs
        diff = abs(out-target).reshape(len(out),img_size,img_size).data.numpy()                                     # Calculate difference between original and output images
        acc = numpy.array([[len(var[numpy.where((i<=var)&(var<i+step_size))]) for var in diff] for i in acc_size])  # Find how many pixels are found in each discrepancy range 
        acc = acc/img_size**2*100                                                                                   # Convert the values to percentages
        results[epoch] = numpy.mean(acc,axis=1)                                                                     # Calculate mean percentage accross all images
    plt.style.use('seaborn')                                                                                      # Set seaborn style
    fig = plt.figure(figsize=(10,6),dpi=80)                                                                       # Initialize figure
    ax1 = fig.add_axes([0.10,0.10,0.83,0.69]) # Main plot
    ax2 = fig.add_axes([0.95,0.10,0.03,0.69]) # Colorbar
    ax3 = fig.add_axes([0.10,0.82,0.83,0.15],sharex=ax1) # Histogram
    img = ax1.imshow(results.T[::-1],aspect='auto',cmap='summer',extent=[0,epochs,0,max_diff])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Discrepancy threshold')
    plt.colorbar(img,label='Percentage of pixels',cax=ax2)                                                        # Plot colorbar
    y = [sum(results[i]) for i in range(epochs)]                                                                  # Sum all percentages for each epoch
    x = numpy.arange(epochs)
    ax3.bar(x,y,width=1,align='edge',color='lightgrey')
    ax3.set_facecolor('white')
    ax3.set_ylim(min(y)-1,max(y)+1)
    ax3.set_title('Reconstruction accuracy')
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.show()

def label_2d_latent(model,data_loader,embeddings=False):
    """
    Scatter plot of 2D latent space with label-based color schemme.

    Parameters
    ----------
    model : :py:class:`torch.nn.Module`
      Trained model.
    data_loader : :py:class:`torch.utils.data.DataLoader`
      Input data to evaluate the trained model with.
    embeddings : :py:class:`bool`
      Whether to display the embeddings or not. 

    """
    model.eval()
    plt.style.use('seaborn')
    fig, ax = plt.subplots(dpi=100)
    for batch_idx, (data,target) in enumerate(data_loader):
        data = data.float()
        z, recon_batch, mu, logvar = model(data.view(-1,numpy.prod(data.shape[-2:])))
        z = z.data.cpu().numpy()
        plt.scatter(z[:,0],z[:,1],s=10,c=target,cmap='cool',alpha=0.5)
        if embeddings:
            for i,img in enumerate(data):
                imagebox = OffsetImage(data[i,0], zoom=0.4)
                ab = AnnotationBbox(imagebox, (z[i,0], z[i,1]),frameon=False)
                ax.add_artist(ab)
    plt.xlabel('Latent variable 1')
    plt.ylabel('Latent variable 2')
    plt.tight_layout()

def success_rate(model,target,img_size,discrepancy_threshold,success_threshold=70):
    """
    Number of reconstructed images for which 70% (or the percentage defined by the success_threshold argument) of the pixels are 90% similar.
    """
    # Set model to evaluation mode
    model.eval()
    # Execute trained model to data
    out, _ = model(target.float())
    # Loop over all output data
    for i in range(len(out)):
        # Normalized outputs
        out[i][0] = (out[i][0]-out[i][0].min())/(out[i][0].max()-out[i][0].min())
    # Calculate difference between original and output images
    diff = abs(out-target).reshape(len(out),img_size,img_size).data.numpy()
    acc = numpy.array([len(var[numpy.where(var<discrepancy_threshold)]) for var in diff])
    acc = acc/img_size**2*100
    # Calculate success rate
    success_rate = sum(i>success_threshold for i in acc)/len(acc)*100
    # Display the following:
    #   - Success rate
    #   - Success threshold above which a single image is considered to be well reconstructed
    #   - Display reconstruction threshold (1 minus discrepancy threshold) above which a single
    #     pixel is considered to be well reconstructed
    print('%.2f%% of the images have'%success_rate,
          '%i%% of their pixels with'%success_threshold,
          '%i%% reconstruction fidelity'%((1-discrepancy_threshold)*100))
    return out,acc
