#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# The executor's wire: the engine contract
# (tensorrt_llm/contracts/engine/v1/engine.proto) and its native codec, compiled
# into the bindings module (tensorrt_llm.bindings.internal.engine_wire).
# protobuf and abseil are linked statically, their symbols kept inside the
# module. Sets ENGINE_WIRE_SOURCES and ENGINE_WIRE_INCLUDE_DIRS for the bindings
# module, which depends on the engine_wire_proto target for the generated C++.

function(engine_wire_fetch_protobuf)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(ABSL_ENABLE_INSTALL OFF)
  set(ABSL_BUILD_TESTING OFF)
  set(protobuf_BUILD_TESTS OFF)
  set(protobuf_INSTALL OFF)
  set(protobuf_BUILD_SHARED_LIBS OFF)
  set(protobuf_WITH_ZLIB OFF)
  set(protobuf_ABSL_PROVIDER module)
  FetchContent_MakeAvailable(abseil-cpp protobuf)
endfunction()
engine_wire_fetch_protobuf()

get_filename_component(contracts ${CMAKE_CURRENT_SOURCE_DIR}/../../contracts
                       ABSOLUTE)
set(proto engine/v1/engine.proto)
set(generated ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(outputs ${generated}/engine/v1/engine.pb.cc
            ${generated}/engine/v1/engine.pb.h)
add_custom_command(
  OUTPUT ${outputs}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${generated}
  COMMAND $<TARGET_FILE:protobuf::protoc> -I ${contracts} --cpp_out=${generated}
          ${contracts}/${proto}
  DEPENDS ${contracts}/${proto} protobuf::protoc
  COMMENT "Generating C++ for ${proto}")
# A custom command's outputs get a rule only in this directory
add_custom_target(engine_wire_proto DEPENDS ${outputs})

set(ENGINE_WIRE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/engineWire.cpp
    ${generated}/engine/v1/engine.pb.cc
    PARENT_SCOPE)
set(ENGINE_WIRE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR} ${generated}
    PARENT_SCOPE)
