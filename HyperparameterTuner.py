class HyperparameterTuner:
    def __init__(self, X, y, num_classes=3):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.best_params = {
            'timesteps': 100,
            'layers': 1,
            'neurons': 100,
            'batch_size': 512,
            'learning_rate': 0.001,
            'activation_ratio': 0.9
        }
        
    def tune_sequentially(self):
        """Sequential tuning as described in the paper"""
        self._tune_layers()
        self._tune_neurons()
        self._tune_batch_size()
        self._tune_activation_ratio()
        return self.best_params

    def _tune_layers(self):
        print("Tuning number of layers...")
        best_acc = 0
        for n_layers in [1, 2, 3]:
            acc = self._cross_validate({'layers': n_layers})
            if acc > best_acc:
                best_acc = acc
                self.best_params['layers'] = n_layers
        print(f"Best layers: {self.best_params['layers']} (acc: {best_acc:.2%})")

    def _tune_neurons(self):
        print("\nTuning number of neurons...")
        best_acc = 0
        for neurons in [10, 50, 100, 150, 200, 250]:
            acc = self._cross_validate({'neurons': neurons})
            if acc > best_acc:
                best_acc = acc
                self.best_params['neurons'] = neurons
        print(f"Best neurons: {self.best_params['neurons']} (acc: {best_acc:.2%})")

    def _tune_batch_size(self):
        print("\nTuning batch size...")
        best_acc = 0
        for batch_size in [32, 64, 128, 256, 512]:
            acc = self._cross_validate({'batch_size': batch_size})
            if acc > best_acc:
                best_acc = acc
                self.best_params['batch_size'] = batch_size
        print(f"Best batch: {self.best_params['batch_size']} (acc: {best_acc:.2%})")

    def _tune_activation_ratio(self):
        print("\nTuning activation ratio...")
        best_acc = 0
        for ratio in [0.6, 0.7, 0.8, 0.9, 1.0]:
            acc = self._cross_validate({'activation_ratio': ratio})
            if acc > best_acc:
                best_acc = acc
                self.best_params['activation_ratio'] = ratio
        print(f"Best ratio: {self.best_params['activation_ratio']} (acc: {best_acc:.2%})")

    def _cross_validate(self, params):
        """10-fold cross-validation with early stopping"""
        kfold = KFold(n_splits=10, shuffle=True)
        accuracies = []
        
        current_params = {**self.best_params, **params}
        
        for train_idx, val_idx in kfold.split(self.X):
            # Build model
            model = Sequential()
            for _ in range(current_params['layers']):
                model.add(RNN(
                    HybridLSTMCell(current_params['neurons'], 
                                 ratio=current_params['activation_ratio']),
                    input_shape=(current_params['timesteps'], self.X.shape[2])
                ))
            model.add(Dense(self.num_classes, activation='softmax'))
            
            # Compile
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=current_params['learning_rate']
                ),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train with early stopping
            history = model.fit(
                self.X[train_idx], self.y[train_idx],
                validation_data=(self.X[val_idx], self.y[val_idx]),
                epochs=1000,
                batch_size=current_params['batch_size'],
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=100,
                    restore_best_weights=True
                )],
                verbose=0
            )
            
            # Store best validation accuracy
            accuracies.append(max(history.history['val_accuracy']))
        
        return np.mean(accuracies)

