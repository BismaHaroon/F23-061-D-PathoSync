import React from "react";
import tw from "twin.macro";
import { css } from "styled-components/macro";

const Overlay = tw.div`fixed top-0 left-0 w-full h-full bg-black opacity-50 z-50`;
const Dialog = tw.div`fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white p-8 rounded z-50`;

const LoadingDialog = ({ message }) => {
  return (
    <>
      <Overlay />
      <Dialog>
        <div css={tw`flex items-center justify-center space-x-4`}>
          <svg
            css={tw`animate-spin h-6 w-6 text-primary-500`}
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              css={tw`opacity-25`}
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              css={tw`opacity-75`}
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0c4.418 0 8 3.582 8 8s-3.582 8-8 8a8.011 8.011 0 01-5.657-2.343"
            ></path>
          </svg>
          <span css={tw`text-primary-500`}>{message}</span>
        </div>
      </Dialog>
    </>
  );
};

export default LoadingDialog;
