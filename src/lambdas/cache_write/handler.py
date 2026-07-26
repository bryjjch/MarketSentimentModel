"""Cache-write Lambda: sole owner of DynamoDB ``sentiment_cache`` writes.

Consumes cache-write tasks from SQS (batch size up to 10, partial batch failures).
Both producers — the pipeline predict Lambda and the on-demand sentiment API — go
through this queue, so this is the only role with write access to the table.

The put is conditional on ``updated_at`` not regressing, so standard-queue
reordering and redeliveries can never overwrite a symbol with older data.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

from finsense_shared import build_cache_item
from finsense_shared.messages import TASK_CACHE_WRITE, validate_task
from finsense_shared.sqs import batch_failures, iter_records

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TABLE_NAME = os.environ["TABLE_NAME"]
_table = boto3.resource("dynamodb").Table(TABLE_NAME)


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    failed: list[str] = []
    written = 0
    skipped_stale = 0

    for message_id, body in iter_records(event):
        try:
            task = validate_task(body, TASK_CACHE_WRITE)
            item = build_cache_item(task)
            _table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(updated_at) OR updated_at <= :u",
                ExpressionAttributeValues={":u": item["updated_at"]},
            )
            written += 1
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                skipped_stale += 1
            else:
                logger.exception("cache_put_failed message_id=%s: %s", message_id, e)
                failed.append(message_id)
        except Exception as e:  # noqa: BLE001 -- one bad message must not fail the batch
            logger.exception("cache_write_failed message_id=%s: %s", message_id, e)
            failed.append(message_id)

    logger.info("cache_write_complete written=%d skipped_stale=%d failed=%d", written, skipped_stale, len(failed))
    return batch_failures(failed)
