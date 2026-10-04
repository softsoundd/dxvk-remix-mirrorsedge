#include <iostream>
#include <string>
#include <vector>
#include <thread>

#include "config/config.h"
#include "config/global_options.h"
#include "util_commands.h"
#include "util_circularqueue.h"
#include "util_atomiccircularqueue.h"


using namespace std;
using namespace Commands;
using namespace bridge_util;

const int QUEUE_SIZE = 5;
const int MEM_SIZE = 640;
void* gMemoryData = NULL;

class CommandQueueHistoryTest {
public:
  static void run() {
    cout << "Begin CommandHistoryQueue smoke test" << endl;
    test_smoke();
    cout << "CommandHistoryQueue successfully smoke tested" << endl;
  }

private:
  static void test_smoke() {
    gMemoryData = new char[MEM_SIZE];
    AtomicCircularQueue<Header, Accessor::Writer> commandQueueObject("Client2ServerCommand",
                                gMemoryData,
                                MEM_SIZE,
                                QUEUE_SIZE);

    // Pushing list of commands into the queue
    if (commandQueueObject.push({ Bridge_Syn, 0, 0, 0 }) != Result::Success) {
      throw string("Issue sending command to the queue");
    }
    if (commandQueueObject.push({ Bridge_Ack, 0, 0, 0 }) != Result::Success) {
      throw string("Issue sending command to the queue");
    }
    if (commandQueueObject.push({ IDirect3DDevice9Ex_GetDeviceCaps, 0, 0, 0 }) != Result::Success) {
      throw string("Issue sending command to the queue");
    }

    // Pulling commands from the queue
    Result result = Result::Failure;
    Header pullResult = commandQueueObject.pull(result);
    if (result != Result::Success) {
      throw string("Issue retrieving command from the queue");
    }
    if (pullResult.command != Bridge_Syn) {
      throw string("Retrieved command from the queue is not as expected");
    }

    // Check if Command sent to the queue is consistent
    vector<D3D9Command> commandSent;
    commandSent = commandQueueObject.getWriterQueueData(3);
    if (commandSent[0] != IDirect3DDevice9Ex_GetDeviceCaps || commandSent[1] != Bridge_Ack || commandSent[2] != Bridge_Syn) {
      throw string("Commands sent do not match");
    }

    // Check if Command recieved from the queue is consistent
    vector<D3D9Command> commandReceived;
    commandReceived = commandQueueObject.getReaderQueueData(1);
    if (commandReceived[0] != Bridge_Syn) {
      throw string("Commands received do not match");
    }
  }
};

// One producer and one consumer thread with their own queue objects over the same memory, as the
// client and server use it, through a queue small enough to wrap and fill constantly.
class CommandQueueThreadedTest {
public:
  static void run() {
    cout << "Begin CommandQueue threaded test" << endl;
    test_threaded();
    cout << "CommandQueue threaded test passed" << endl;
  }

private:
  static void test_threaded() {
    static constexpr size_t kQueueSize = 7;
    static constexpr uint32_t kCount = 1'000'000;
    using Writer = AtomicCircularQueue<Header, Accessor::Writer>;
    using Reader = AtomicCircularQueue<Header, Accessor::Reader>;

    const size_t memSize = Writer::getExtraMemoryRequirements() + sizeof(Header) * kQueueSize;
    vector<char> memory(memSize);
    Writer writer("ThreadedTestCommand", memory.data(), memSize, kQueueSize);
    Reader reader("ThreadedTestCommand", memory.data(), memSize, kQueueSize);

    thread producer([&writer] {
      for (uint32_t i = 0; i < kCount; ++i) {
        writer.push({ static_cast<D3D9Command>(i & 0x7FFF), 0, i, ~i });
      }
    });

    // Drains every command even after a mismatch, or the producer would block on a full queue.
    string error;
    for (uint32_t i = 0; i < kCount; ++i) {
      Result result = Result::Failure;
      if ((i & 1) == 0) {
        const Header& peeked = reader.peek(result);
        if (error.empty() && (result != Result::Success || peeked.dataOffset != i)) {
          error = "Peeked command out of order";
        }
      }
      const Header h = reader.pull(result);
      if (error.empty() && (result != Result::Success || h.dataOffset != i || h.pHandle != ~i)) {
        error = "Pulled command out of order";
      }
    }
    producer.join();

    if (!error.empty()) {
      throw error;
    }
    if (!reader.isEmpty()) {
      throw string("Queue not empty after the last command");
    }
  }
};

int main() {
  try {
    CommandQueueHistoryTest::run();
    CommandQueueThreadedTest::run();
  }
  catch (const string& errorMessage) {
    cerr << errorMessage << endl;
	return -1;
  }
  delete[] gMemoryData;
  return 0;
}

