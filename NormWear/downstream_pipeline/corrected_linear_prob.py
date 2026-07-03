
# ML experimenting
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import numpy as np
# from scipy import stats

from sklearn.multioutput import MultiOutputRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor, Ridge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization,Dropout,Conv1D,MaxPool1D,Flatten,Reshape
from tensorflow.keras.activations import softmax
# from sklearn.preprocessing import MinMaxScaler

def linear_prob(args,
    x_train,
    y_train,
    x_test,
    y_test,
    task_type='class',
    random_state=42
):
    tf.keras.utils.set_random_seed(random_state)
    steps_per_epoch = max(1, len(x_train) // args.batch_size)

    wandb.init(
    # set the wandb project where this run will be logged
    project="NormWear full shot",

    # track hyperparameters and run metadata with wandb.config
    config={
        "number_of_layer": args.number_of_layer,
        "metric":  args.metric,
        "data":"WESAD",
        "epoch": args.epochs,
        "batch_size": args.batch_size,
        "regularization_parameter": args.regularization_parameter,
        "learning_rate_schedule": args.lr_scheduler,
        "initial_learning_rate": args.initial_learning_rate,
        "decay_steps": steps_per_epoch,
        "decay_rate": args.decay_rate,
        "staircase":False,
        "validation_split":0.2,
        "patience": args.patience,
        #"seed":23,
        "dropout_rate": args.dropout_rate
    }
    )
    config =wandb.config

    # init learning rate schedule / optimizer. Some schedulers (plateau) can't be
    # baked into a tf.keras LR schedule object since they react to validation
    # metrics at runtime, so they're implemented as a callback instead.
    scheduler_callbacks = []
    if config.learning_rate_schedule == 'exponential':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.initial_learning_rate,
            decay_steps=config.decay_steps,
            decay_rate=config.decay_rate, staircase=config.staircase)
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
    elif config.learning_rate_schedule == 'cosine':
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.initial_learning_rate,
            decay_steps=config.epoch * steps_per_epoch)
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
    elif config.learning_rate_schedule == 'plateau':
        optimizer = tf.keras.optimizers.Adam(config.initial_learning_rate)
        scheduler_callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=config.decay_rate,
            patience=max(1, config.patience // 2), min_lr=1e-7))
    elif config.learning_rate_schedule == 'constant':
        optimizer = tf.keras.optimizers.Adam(config.initial_learning_rate)
    else:
        raise ValueError("Unknown lr_scheduler: {}".format(config.learning_rate_schedule))

    if task_type == 'class':
        lp = LogisticRegression(
            max_iter=500,
            solver='newton-cg',
            # solver='sag',
            # penalty=None,
            # C=1e6, # very strong
            C=1e0, # so so
            # C=2e1, # very weak
            # C=20e1, # very weak
            # random_state=random_state,
            # class_weight='balanced'
        )
        lp = Sequential()
        print(config.regularization_parameter)
        for i in range(0, config.number_of_layer - 2):
            lp.add(Dense(units=2**(config.number_of_layer-i), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config.regularization_parameter)))
            lp.add(BatchNormalization())
            lp.add(Dropout(config.dropout_rate))
        lp.add(Dense(units=len(set(y_train)), activation="linear", kernel_regularizer=tf.keras.regularizers.l2(config.regularization_parameter)))
        lp.build(input_shape=(None, x_train.shape[1]))
        print("lp.summary():", lp.summary())
        lp.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        #lp.load_weights('/home/user/Bureau/NormWear/tmp/checkpoint.model_1000.keras')

        # print(set(y_train))
    else:
        # lp = LinearRegression()
        #lp = Ridge(
        #    max_iter=500,
        #    solver="cholesky",
        #    alpha=1e0
        # alpha=1e1)
        # lp = SGDRegressor(
        #     max_iter=1000,
        #     # penalty='elasticnet',
        #     # learning_rate='optimal',
        #     # alpha=1e-2,
        #     # tol=1e-4,
        #     # loss='huber',
        #     # epsilon=1e-6
        # )
        # lp = Lasso()
        output_dim = y_test.shape[1] if len(y_test.shape) > 1 and y_test.shape[1] > 1 else 1
        lp = Sequential([
            Dropout(config.dropout_rate),
            BatchNormalization(),
            Dense(units=output_dim, activation="linear", kernel_regularizer=tf.keras.regularizers.l2(config.regularization_parameter)),
        ])

        lp.build(input_shape=(None, x_train.shape[1]))
        print("lp.summary():", lp.summary())
        lp.compile(optimizer=optimizer,
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
        # # z normalize
        # y_mean, y_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
        # y_train = (y_train - y_mean) / y_std
        # y_test = (y_test - y_mean) / y_std

        # # min-max normalize by y_train, so the output keep consistent
        # scaler = MinMaxScaler(feature_range=(1, 10))
        # if len(y_train.shape) < 2:
        #     y_train = np.reshape(y_train, (-1, 1))
        #     y_test = np.reshape(y_test, (-1, 1))
        # scaler.fit(y_train)
        # y_train = scaler.transform(y_train)
        # y_test = scaler.transform(y_test)

        # log scale
        # print(np.isnan(np.log(y_train+1)).sum() / len(y_train))
        # print(np.isnan(np.log(y_train+1)).sum(), len(y_train))
        # print(np.isnan(np.log(y_test+1)).sum() / len(y_test))
        # print(np.isnan(np.log(y_test+1)).sum(), len(y_test))
        # exit()

        y_train = np.nan_to_num(np.log(y_train+1))
        y_test = np.nan_to_num(np.log(y_test+1))
    
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)

    # # shuffle
    # indices = np.arange(x_train.shape[0])
    # np.random.shuffle(indices)
    # x_train = x_train[indices]
    # y_train = y_train[indices]
    
    # fit linear model
    # start = time.time()
    # print("Fitting Linear Model...")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./tmp/checkpoint.model_{epoch:02d}.keras',
        monitor='mae' if task_type != 'class' else 'accuracy',
        mode='min' if task_type != 'class' else 'max',
            save_freq='epoch',)
    history = lp.fit(x_train, y_train,epochs=config.epoch,verbose=2, batch_size=config.batch_size, callbacks=[model_checkpoint_callback,WandbMetricsLogger(log_freq="epoch"),WandbModelCheckpoint("model.keras")]+scheduler_callbacks,validation_split=config.validation_split)
    wandb.finish()
    # end = time.time()
    # print("Time consumed:", end-start, "s")

    if task_type != 'class':
        best_epoch = int(np.argmin(history.history['val_loss']))
        print("BEST_VAL_MAE: {:.6f} (train_mae: {:.6f}, best_epoch: {}/{})".format(
            history.history['val_mae'][best_epoch], history.history['mae'][best_epoch],
            best_epoch + 1, len(history.history['loss'])))

    # test time
    return calculate_score(lp, x_test, y_test, task_type, y_train=y_train)

def calculate_score(lp, x_test, y_true, task_type, y_train=None):
    if task_type == "reg":
        y_pred = lp.predict(x_test)

        # filterout nan
        nan_to_mean = np.isnan(y_pred) * np.mean(y_train, axis=0)
        y_pred = np.nan_to_num(y_pred) + nan_to_mean

        # ### simple-mean ###
        # y_mean = np.mean(y_train, axis=0)
        # y_pred = np.array([y_mean for _ in range(len(y_true))])

        # numerical stability
        y_pred = np.clip(y_pred, np.min(y_train), np.max(y_train))
        # if len(y_pred.shape) > 1 and y_pred.shape[1] < 2:
        #     y_pred = y_pred[:, 0]

        # print(np.mean(y_pred), np.min(y_pred), np.max(y_pred))

        final_scores = [1 - np.mean(np.absolute((y_true - y_pred) / y_true))]
        return final_scores
    else:
        y_pred = softmax(lp.predict(x_test))

        final_scores = list()

        # roc auc
        y_set = list(set(y_true))
        y_pred_class = np.argmax(y_pred, axis=1)

        # ### simple-mode ###
        # mode = stats.mode(y_train).mode
        # y_pred_class = np.array([mode for _ in range(len(y_true))])
        # one_hot = np.zeros(y_pred.shape)
        # one_hot[np.arange(len(y_pred)), y_pred_class] = 1
        # y_pred = one_hot

        # remove non-exist class in y_true
        #y_pred = y_pred[:, y_set]
        y_pred /= np.sum(y_pred, axis=1, keepdims=True)

        # calculate score
        if len(y_set) <= 2:
            final_scores.append(roc_auc_score(y_true, y_pred[:, 1]))
            final_scores.append(average_precision_score(y_true, y_pred[:, 1]))
        else:
            final_scores.append(roc_auc_score(y_true, y_pred, multi_class="ovo", average="macro", labels=y_set))
            final_scores.append(average_precision_score(y_true, y_pred, average="macro"))
        
        final_scores.append(np.mean((y_true == y_pred_class)))
        final_scores.append(precision_score(y_true, y_pred_class, average='macro'))
        final_scores.append(recall_score(y_true, y_pred_class, average='macro'))
        final_scores.append(f1_score(y_true, y_pred_class, average="macro"))
        
        # final_scores.append(balanced_accuracy_score(y_true, y_pred_class))
        
        return final_scores