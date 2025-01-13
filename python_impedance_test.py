import argparse
import time
import logging
import numpy as np
import math
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes, DetrendOperations

# The sampling rate(125) and the number of channels(13) are hardcoded for various tests and data checks.


# stack class
class FixedStack:
    def __init__(self, size=125):  # 125 is sample rate
        self.max_size = size  # Maximum size of the stack
        self.stack = []  # List

    def setSize(self, size):
        # maximum size of the stack
        self.max_size = size

    def fill(self, obj):
        # Fill the stack
        for _ in range(self.max_size):
            self.push(obj)

    def push(self, obj):
        # Add a new element to the stack
        while len(self.stack) >= self.max_size:
            self.stack.pop(0)  # Remove the oldest element if full
        self.stack.append(obj)
        return obj

    def size(self):
        # Return the current size of the stack
        return len(self.stack)

    def __getitem__(self, index):
        # Access an element by index
        return self.stack[index]

    def __str__(self):
        # Return a string representation with a preview of recent data
        preview = self.stack[-10:]
        return f"Buffer size: {len(self.stack)}, Data preview: {preview}"

    def __repr__(self):
        # Alias __str__ for representation
        return self.__str__()


# get data from stack
def getdata(accumulated_data, max_samples):
    end_index = accumulated_data.size()  # 22 * 125 = 2,750
    start_index = max(0, end_index - max_samples)  # 2,750 - 2,750 = 0
    # return accumulated_data[ 0 : 2750 ]
    return accumulated_data[start_index:end_index]


# Main function
def main():
    # Enable debug logging
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, required=False, default=0)
    parser.add_argument("--ip-port", type=int, required=False, default=0)
    parser.add_argument("--ip-protocol", type=int, required=False, default=0)
    parser.add_argument("--ip-address", type=str, required=False, default="")
    parser.add_argument("--serial-port", type=str, required=False, default="COM3")
    parser.add_argument("--mac-address", type=str, required=False, default="")
    parser.add_argument("--other-info", type=str, required=False, default="")
    parser.add_argument("--streamer-params", type=str, required=False, default="")
    parser.add_argument("--serial-number", type=str, required=False, default="")
    parser.add_argument(
        "--board-id", type=int, required=False, default=BoardIds.CYTON_DAISY_BOARD
    )
    parser.add_argument("--file", type=str, required=False, default="")
    parser.add_argument(
        "--master-board", type=int, required=False, default=BoardIds.NO_BOARD
    )
    args = parser.parse_args()

    # board parameters
    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    # Initialize the board
    board = BoardShim(args.board_id, params)

    # Create a buffer for accumulating data
    accumulated_data = FixedStack()
    accumulated_data.setSize(22 * 125)  # size
    accumulated_data.fill([0.0] * 13)  # Pre-fill with zeros

    board.prepare_session()  # Prepare the session

    # Command

    command = "x1000100Xz101Z"
    # command = "x1000100Xz101Zx2060110Xx3060110Xx4060110Xx5060110Xx6060110Xx7060110Xx8060110XxQ060110XxW060110XxE060110XxR060110XxT060110XxY060110XxU060110XxI060110X"

    response = board.config_board(command)
    print("response:", response)
    # response: Success: Channel set for 1$$$Success: Lead off set for 1$$$

    board.start_stream()  # Start data stream from the board

    # Buffers for raw and filtered data
    dataProcessingRawBuffer = [[0.0 for _ in range(22 * 125)] for _ in range(13)]
    dataProcessingFilteredBuffer = [[0.0 for _ in range(22 * 125)] for _ in range(13)]

    acc_data = []
    data_std_uV = [0] * 13  # Standard deviation for each channel
    data_elec_imp_ohm = [0] * 13  # Impedance for each channel

    # Main loop for processing EEG data
    while True:
        data = board.get_board_data()  # get data from the board

        # print("data[1]:", len(data[1]))
        # print("data", data[1:4, -5:])  # Channel 1: O2, Channel 2: O1, Channel 3: T5

        # Push new entries to the buffer
        for i in range(len(data[1])):
            new_entry = []
            for j in range(13):
                new_entry.append(data[j + 1][i])
            # print("new_entry:", new_entry)
            accumulated_data.push(new_entry)

        # Extract recent data for processing
        current_data = getdata(
            accumulated_data, 22 * 125
        )  # 22(dataBuff_len_sec) * sampleing rate.

        # Update raw and filtered buffers
        for j in range(13):
            for i in range(22 * 125):
                dataProcessingRawBuffer[j][i] = float(current_data[i][j])
            dataProcessingFilteredBuffer[j] = np.copy(dataProcessingRawBuffer[j])

        # Apply filters to each chnnel data
        for j in range(13):

            temp_array = np.array(dataProcessingFilteredBuffer[j])

            # DataFilter.detrend(temp_array, DetrendOperations.CONSTANT.value)

            DataFilter.perform_bandstop(
                temp_array, 125, 58, 62, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
            )
            DataFilter.perform_bandpass(
                temp_array, 125, 5, 50, 2, FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
            )
            DataFilter.remove_environmental_noise(temp_array, 125, NoiseTypes.FIFTY)
            DataFilter.remove_environmental_noise(temp_array, 125, NoiseTypes.SIXTY)

            acc_data = temp_array
            # last sample rate data 125
            foo_data_filt = acc_data[-int(125) :]
            # Calculate each channel data_std_uV
            data_std_uV[j] = np.std((foo_data_filt))

            # if j == 0:
            #     print("data_std_uV[0] : ", data_std_uV[0])

        # Calculate impedance for each channel
        for j in range(13):

            impedance = (math.sqrt(2.0) * (data_std_uV[j]) * 1.0e-6) / 6.0e-9
            impedance -= 2200.0

            if impedance < 0.0:
                impedance = 0.0

            data_elec_imp_ohm[j] = impedance

            # if j == 0:
            #     print("data_elec_imp_ohm[0]: ", data_elec_imp_ohm[0])

        # Print impedance for the first channel
        print("impedance in kΩ: ", data_elec_imp_ohm[0] / 1000)

        time.sleep(1)  # Sleep for 1 second

    # Stop the stream and release resources
    # board.stop_stream()
    # board.release_session()


if __name__ == "__main__":
    main()
