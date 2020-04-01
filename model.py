def TextCNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test):
    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    main_input = Input(shape=(300,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    #embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = GlobalMaxPooling1D()(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = GlobalMaxPooling1D()(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = GlobalMaxPooling1D()(cnn3)
    # 合并三个模型的输出向量
    cnn = Concatenate()([cnn1, cnn2, cnn3])
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(43, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
    one_hot_labels = to_categorical(y_train, num_classes=43)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=5)
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))
        
        
        
TextCNN_model(x_train, y_train, x_test, y_test)
