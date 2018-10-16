class config:
    batch_size = 1
    num_epochs = 1000
    lambda_L = 5
    lambdas_j = [-1e-6/3, -1e-6/3, -1e-6/3, -1e-6/3]
    beta1 = 0.9
    label_smooth = 0.9  # 0.85 # like in https://github.com/rick-chang/OneNet
    learning_rate = 2e-4
    ckpt_dir = "../logs/"
    debug = False
    save_interval = 5
