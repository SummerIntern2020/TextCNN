def TextCNN(num_classes, seq_length, embedding_dims, filter_sizes, num_filters, last_activation = 'softmax',l2_reg_lambda = 0.0,model_img_path=None):
        input = Input((seq_length,))
        logging.info("x_input.shape: %s" % str(x_input.shape))
        embedding = Embedding(imput_dim = num_classes, output_dim = embedding_dims, input_length = seq_length)(input)
        convs = []
        for kernel_size in filter_sizes:
            C = Conv1D(128, kernel_size, activation='relu')(embedding)
            M = GlobalMaxPooling1D()(C)
            convs.append(M)
            logging.info("kernel_size: %s \t c.shape: %s \t p.shape: %s" % (kernel_size, str(C.shape), str(M.shape)))
        convs = concatenate([p for p in convs])
        logging.info("pool_output.shape: %s" % str(convs.shape))
        flat = Flatten()(convs)
        dropping = Dropout(0.2)(flat)
        y = Dense(output_dim, activation='softmax')(dropping)
        model = Model([x_input], outputs=[y])
        if model_img_path:
            plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
        model.summary()
        return model

        output = Dense(num_classes, activation=last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model
