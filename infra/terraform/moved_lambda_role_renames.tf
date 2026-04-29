moved {
  from = aws_lambda_function.predict
  to   = aws_lambda_function.api_inference
}

moved {
  from = aws_lambda_function.prediction
  to   = aws_lambda_function.ingestion_prediction
}

moved {
  from = aws_lambda_function.sentiment
  to   = aws_lambda_function.api_sentiment_by_symbol
}

moved {
  from = aws_iam_role.predict_lambda
  to   = aws_iam_role.api_inference_lambda
}

moved {
  from = aws_iam_role_policy_attachment.predict_lambda_basic
  to   = aws_iam_role_policy_attachment.api_inference_lambda_basic
}

moved {
  from = aws_iam_role_policy.predict_lambda_invoke
  to   = aws_iam_role_policy.api_inference_lambda_invoke
}

moved {
  from = aws_iam_role.prediction_lambda
  to   = aws_iam_role.ingestion_prediction_lambda
}

moved {
  from = aws_iam_role_policy_attachment.prediction_lambda_basic
  to   = aws_iam_role_policy_attachment.ingestion_prediction_lambda_basic
}

moved {
  from = aws_iam_role_policy.prediction_lambda
  to   = aws_iam_role_policy.ingestion_prediction_lambda
}

moved {
  from = aws_iam_role.sentiment_lambda
  to   = aws_iam_role.api_sentiment_by_symbol_lambda
}

moved {
  from = aws_iam_role_policy_attachment.sentiment_lambda_basic
  to   = aws_iam_role_policy_attachment.api_sentiment_by_symbol_lambda_basic
}

moved {
  from = aws_iam_role_policy.sentiment_lambda_invoke
  to   = aws_iam_role_policy.api_sentiment_by_symbol_lambda_invoke
}

moved {
  from = aws_iam_role_policy.sentiment_lambda_secrets
  to   = aws_iam_role_policy.api_sentiment_by_symbol_lambda_secrets
}

moved {
  from = aws_apigatewayv2_integration.predict
  to   = aws_apigatewayv2_integration.api_inference
}

moved {
  from = aws_lambda_permission.apigw_invoke
  to   = aws_lambda_permission.apigw_invoke_api_inference
}

moved {
  from = aws_apigatewayv2_integration.sentiment
  to   = aws_apigatewayv2_integration.api_sentiment_by_symbol
}

moved {
  from = aws_lambda_permission.apigw_invoke_sentiment
  to   = aws_lambda_permission.apigw_invoke_api_sentiment_by_symbol
}
