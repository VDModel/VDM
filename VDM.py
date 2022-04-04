
from tensorflow import keras
from keras.layers import Input, Dense, TimeDistributed, Concatenate, GRU
import keras.backend as K
from keras import regularizers, Model
import sklearn

dynamic_train = []
dynamic_valid = []
dynamic_test = []
static_train = []
static_valid = []
static_test = []
y_train = []
y_valid = []
y_test = []

def det_coeff(y_true, y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return K.ones_like(v) - (u / v)

def create_model():
    dynamic_size = 4
    static_size = 28
    gru_size = 32
    l2_regular = regularizers.l2(1e-3)

    input_dynamic = Input(shape=(None, dynamic_size), name='input_dynamic')
    input_static = Input(shape=(None, static_size), name='input_statics')

    gru = GRU(gru_size, activation='tanh', recurrent_activation='sigmoid', name='gru', kernel_regularizer=l2_regular,
              return_sequences=True)(input_dynamic)
    concat = Concatenate(name='concat')([gru, input_static])
    dense = TimeDistributed(Dense(128, activation='relu'), name='dense')(concat)
    output = TimeDistributed(Dense(1, activation='tanh'), name='output')(dense)
    model = Model([input_dynamic, input_static], output)
    opt = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', det_coeff])
    return model

print(create_model().summary())

model = create_model()
epoch = 400
batch_size = 1000
save_dir = r'./trained_weights/vdm'
save_callback = keras.callbacks.ModelCheckpoint(save_dir, monitor='val_loss', verbose=1, save_best_only=True,
                                                save_weights_only=True, mode='auto', period=1)
model.fit(x=[dynamic_train, static_train], y=y_train, batch_size=batch_size, epochs=epoch, callbacks=[save_callback],
          verbose=1, validation_data=([dynamic_valid, static_valid], y_valid), shuffle=True)
history = model.fit(x=[dynamic_train, static_train], y=y_train, batch_size=batch_size, epochs=epoch,
                    callbacks=[save_callback], verbose=1, validation_data=([dynamic_valid, static_valid], y_valid),
                    shuffle=True)


model_test = create_model()
model_test.load_weights(save_dir)

y_pred = []
y_true = []

y_pred = model_test.predict([dynamic_test[:, :, :], static_test[:, :, :]])
y_true = y_test

y_pred = y_pred.flatten()
y_true = y_true.flatten()

print('Mean Squared Error: %.6f' % sklearn.metrics.mean_squared_error(y_pred=y_pred, y_true=y_true))
print('Mean Absolute Error: %.6f' % sklearn.metrics.mean_absolute_error(y_pred=y_pred, y_true=y_true))
print('R2 Score: %.6f' % sklearn.metrics.r2_score(y_pred=y_pred, y_true=y_true))
adj_r2 = 1 - (1 - sklearn.metrics.r2_score(y_pred=y_pred, y_true=y_true)) * (len(y_pred) - 1) / (len(y_pred) - 5 - 1)
print('Adjusted R2 Score: %.6f' % adj_r2)
