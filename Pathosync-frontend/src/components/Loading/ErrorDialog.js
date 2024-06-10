import React from "react";
import tw from "twin.macro";

const Overlay = tw.div`fixed top-0 left-0 w-full h-full bg-black opacity-50 z-50`;
const Dialog = tw.div`fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white p-8 rounded z-50`;
const Message = tw.div`text-xl text-red-500`;

const ErrorDialog = ({ message, onClose }) => {
  return (
    <>
      <Overlay />
      <Dialog>
        <Message>{message}</Message>
        <div css={tw`flex items-center justify-center mt-4`}>
          <button css={tw`bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600`} onClick={onClose}>Close</button>
        </div>
      </Dialog>
    </>
  );
};

export default ErrorDialog;
