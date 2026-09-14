CHUNK_LIMIT = 50

def sentence_buffer(stream):
  buffer = ""
  chunk_count = 0

  for chunk in stream:

    if chunk:
      buffer += chunk
      chunk_count += 1

      if chunk_count >= CHUNK_LIMIT:
        cut = buffer.rfind(" ")

        if cut != -1:
          yield buffer[:cut]
          buffer = buffer[cut:]
        else:
          yield buffer
          buffer = ""

        chunk_count = 0

  if buffer.strip():
    yield buffer
    

