variable "project_name" {
  type        = string
  description = "Resource name prefix."
}

variable "queues" {
  type = map(object({
    name                       = string
    visibility_timeout_seconds = number
    message_retention_seconds  = number
    max_receive_count          = number
    dlq_retention_seconds      = optional(number, 1209600) # 14 days
  }))
  description = "Logical key -> queue settings. The key is what callers index queue_arns/queue_urls by; name is the suffix appended to project_name."

  default = {
    # pipeline_dispatch -> pipeline_collect (collect Lambda timeout 120s)
    collect = {
      name                       = "collect"
      visibility_timeout_seconds = 720
      message_retention_seconds  = 345600 # 4 days
      max_receive_count          = 3
    }

    # pipeline_collect -> pipeline_predict (predict Lambda timeout 300s)
    predict = {
      name                       = "predict"
      visibility_timeout_seconds = 1800
      message_retention_seconds  = 345600
      max_receive_count          = 3
    }

    # pipeline_predict -> pipeline_label (label Lambda timeout 600s)
    label = {
      name                       = "label"
      visibility_timeout_seconds = 3600
      message_retention_seconds  = 345600
      max_receive_count          = 3
    }

    # pipeline_predict + api_sentiment -> cache_write (cache_write Lambda timeout 30s)
    cache_write = {
      name                       = "cache-write"
      visibility_timeout_seconds = 180
      message_retention_seconds  = 86400 # 1 day; stale sentiment is worthless
      max_receive_count          = 5
    }
  }
}
