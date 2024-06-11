import React from "react";
import tw from "twin.macro";
import { css } from "styled-components/macro";

const Overlay = tw.div`fixed top-0 left-0 w-full h-full bg-black opacity-50 z-50`;
const Dialog = tw.div`fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white p-8 rounded z-50`;

const ErrorDialog = ({ message, onClose }) => {
  return (
    <>
      <Overlay onClick={onClose} />
      <Dialog>
        <div css={tw`flex flex-col items-center justify-center`}>
          <svg
            css={tw`mb-4 h-6 w-6 text-red-500`}
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M18 10c0 4.418-3.582 8-8 8s-8-3.582-8-8 3.582-8 8-8 8 3.582 8 8zM4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
          <span css={tw` text-center`}>{message}</span>
          <button
            css={tw`mt-4 text-white px-6 py-2 rounded cursor-pointer`}
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </Dialog>
    </>
  );
};

export default ErrorDialog;
