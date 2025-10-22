# dependencies

find_package(Threads REQUIRED)

# third-party

if (WHISPER_SDL2)
    # SDL2
    find_package(SDL2 REQUIRED)

    string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

    message(STATUS "SDL2_INCLUDE_DIRS = ${SDL2_INCLUDE_DIRS}")
    message(STATUS "SDL2_LIBRARIES    = ${SDL2_LIBRARIES}")
endif()

# common

set(TARGET common)

unset(COMMON_EXTRA_LIBS)

if (WHISPER_FFMPEG)
    # As of cmake 3.27, there is no official cmake support for FindFFmpeg.
    # Consequnelty we added a FindFFmpeg.cmake script the cmake subfolder:
    # whisper.cpp does not need the full ffmpeg libs, just AVFORMAT AVCODEC AVUTIL SWRESAMPLE
    # libswresample  performs highly optimized audio resampling, rematrixing and sample format conversion operations
    # libavcodec provides a generic encoding/decoding framework and contains multiple decoders and encoders for audio, video and subtitle streams, and several bitstream filters.
    # libavformat provides a generic framework for multiplexing and demultiplexing (muxing and demuxing) audio, video and subtitle streams.
    find_package(FFmpeg REQUIRED)

    if (NOT ${FFMPEG_FOUND})
        message(FATAL_ERROR "Cannot find ffmpeg libs/headers")
    endif()

    message(STATUS "Found ffmpeg libs:       ${FFMPEG_LIBRARIES}")
    message(STATUS "Found ffmpeg headers in: ${FFMPEG_INCLUDE_DIRS}")
    message(STATUS "ffmpeg definitions:      ${FFMPEG_DEFINITIONS}")
    message(STATUS "Found avformat           ${AVFORMAT_VERSION}")

    include_directories(${FFMPEG_INCLUDE_DIRS})
    add_compile_definitions(WHISPER_FFMPEG)

    list(APPEND COMMON_EXTRA_LIBS ${FFMPEG_LIBRARIES})

    set(COMMON_SOURCES_FFMPEG ffmpeg-transcode.cpp)
endif()


add_library(${TARGET} STATIC
    common.h
    common.cpp
    common-ggml.h
    common-ggml.cpp
    common-whisper.h
    common-whisper.cpp
    grammar-parser.h
    grammar-parser.cpp
    ${COMMON_SOURCES_FFMPEG}
    )

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE whisper ${COMMON_EXTRA_LIBS} ${CMAKE_DL_LIBS})

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${TARGET} PROPERTIES FOLDER "libs")

if (WHISPER_SDL2)
    # common-sdl

    set(TARGET common-sdl)

    add_library(${TARGET} STATIC
        common-sdl.h
        common-sdl.cpp
        )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PUBLIC  ${SDL2_INCLUDE_DIRS})
    target_link_libraries     (${TARGET} PRIVATE ${SDL2_LIBRARIES})

    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

# add json lib
add_library(json_cpp INTERFACE)
target_include_directories(json_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
    add_subdirectory(whisper.wasm)
    add_subdirectory(stream.wasm)
    add_subdirectory(command.wasm)
    add_subdirectory(bench.wasm)
    add_subdirectory(wchess)
elseif(CMAKE_JS_VERSION)
    add_subdirectory(addon.node)
else()
    add_subdirectory(cli)
    add_subdirectory(bench)
    add_subdirectory(server)
    add_subdirectory(quantize)
    add_subdirectory(vad-speech-segments)
    if (WHISPER_SDL2)
        add_subdirectory(stream)
        add_subdirectory(command)
        add_subdirectory(talk-llama)
        add_subdirectory(lsp)
        if (GGML_SYCL)
            add_subdirectory(sycl)
        endif()
    endif (WHISPER_SDL2)

    add_subdirectory(deprecation-warning)
endif()

if (WHISPER_SDL2)
    add_subdirectory(wchess)
endif (WHISPER_SDL2)

# Ensure whisper-stream target is built by default with examples
# Match the style of other top-level example targets (whisper-cli, whisper-server)
# If the target already exists from examples/stream, this block is a no-op.
if (WHISPER_SDL2)
    if (NOT TARGET whisper-stream)
        add_executable(whisper-stream stream/stream.cpp)
        include(DefaultTargetOptions)
        target_link_libraries(whisper-stream PRIVATE common common-sdl whisper ${WHISPER_EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})
        set_target_properties(whisper-stream PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        install(TARGETS whisper-stream RUNTIME)
    endif()
endif()
