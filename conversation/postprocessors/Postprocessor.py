
class Postprocessor:
    def postprocess(self, sentence_features, predicted_annotations):
        is_question_index = 8
        topic_position_index = 13
        for sentence_index, features in enumerate(sentence_features):
            if sentence_index > 0:
                if sentence_features[sentence_index-1][is_question_index] == 1:
                    predicted_annotations[sentence_index] = max(predicted_annotations[sentence_index], predicted_annotations[sentence_index-1])
                if sentence_features[sentence_index][topic_position_index] == 0:
                    predicted_annotations[sentence_index] = 1
        return predicted_annotations
