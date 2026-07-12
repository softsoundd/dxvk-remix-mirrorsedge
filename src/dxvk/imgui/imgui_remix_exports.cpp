/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "imgui_remix_exports.h"
#include "imgui.h"
#include "implot.h"
#include <atomic>

// Static callback storage - only one wrapper can register at a time.
static std::atomic<PFN_remixapi_imgui_DrawCallback> s_drawCallback{ nullptr };
static std::atomic<void*> s_drawCallbackUserData{ nullptr };

// --- Callback Registration ---

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_RegisterDrawCallback(
    PFN_remixapi_imgui_DrawCallback callback, void* userData) {
  s_drawCallbackUserData.store(userData, std::memory_order_release);
  s_drawCallback.store(callback, std::memory_order_release);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_UnregisterDrawCallback() {
  s_drawCallback.store(nullptr, std::memory_order_release);
  s_drawCallbackUserData.store(nullptr, std::memory_order_release);
}

void remixapi_imgui_InvokeDrawCallback() {
  auto cb = s_drawCallback.load(std::memory_order_acquire);
  if (cb) {
    cb(s_drawCallbackUserData.load(std::memory_order_acquire));
  }
}

int remixapi_imgui_HasDrawCallback() {
  return s_drawCallback.load(std::memory_order_acquire) != nullptr ? 1 : 0;
}

// --- Windows ---

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_Begin(const char* name, int* p_open, int flags) {
  bool open = p_open ? (*p_open != 0) : true;
  bool result = ImGui::Begin(name, p_open ? &open : nullptr, (ImGuiWindowFlags)flags);
  if (p_open) *p_open = open ? 1 : 0;
  return result ? 1 : 0;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_End() {
  ImGui::End();
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_BeginChild(const char* str_id, float w, float h, int border, int flags) {
  return ImGui::BeginChild(str_id, ImVec2(w, h), border != 0, (ImGuiWindowFlags)flags) ? 1 : 0;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_EndChild() {
  ImGui::EndChild();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_SetNextWindowPos(float x, float y, int cond) {
  ImGui::SetNextWindowPos(ImVec2(x, y), (ImGuiCond)cond);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_SetNextWindowSize(float w, float h, int cond) {
  ImGui::SetNextWindowSize(ImVec2(w, h), (ImGuiCond)cond);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_SetNextWindowCollapsed(int collapsed, int cond) {
  ImGui::SetNextWindowCollapsed(collapsed != 0, (ImGuiCond)cond);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_SetNextWindowFocus() {
  ImGui::SetNextWindowFocus();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_GetWindowSize(float* out_w, float* out_h) {
  ImVec2 sz = ImGui::GetWindowSize();
  if (out_w) *out_w = sz.x;
  if (out_h) *out_h = sz.y;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_GetWindowPos(float* out_x, float* out_y) {
  ImVec2 pos = ImGui::GetWindowPos();
  if (out_x) *out_x = pos.x;
  if (out_y) *out_y = pos.y;
}

// --- Layout ---

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_SameLine(float offset_from_start_x, float spacing) {
  ImGui::SameLine(offset_from_start_x, spacing);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_Separator() {
  ImGui::Separator();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_Spacing() {
  ImGui::Spacing();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_Dummy(float w, float h) {
  ImGui::Dummy(ImVec2(w, h));
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_Indent(float indent_w) {
  ImGui::Indent(indent_w);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_Unindent(float indent_w) {
  ImGui::Unindent(indent_w);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_NewLine() {
  ImGui::NewLine();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_BeginGroup() {
  ImGui::BeginGroup();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_EndGroup() {
  ImGui::EndGroup();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_GetContentRegionAvail(float* out_w, float* out_h) {
  ImVec2 avail = ImGui::GetContentRegionAvail();
  if (out_w) *out_w = avail.x;
  if (out_h) *out_h = avail.y;
}

// --- ID Stack ---

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PushID_Str(const char* str_id) {
  ImGui::PushID(str_id);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PushID_Int(int int_id) {
  ImGui::PushID(int_id);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PopID() {
  ImGui::PopID();
}

// --- Text ---

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_Text(const char* text) {
  ImGui::TextUnformatted(text);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_TextColored(float r, float g, float b, float a, const char* text) {
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(r, g, b, a));
  ImGui::TextUnformatted(text);
  ImGui::PopStyleColor();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_TextWrapped(const char* text) {
  ImGui::PushTextWrapPos(0.0f);
  ImGui::TextUnformatted(text);
  ImGui::PopTextWrapPos();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_BulletText(const char* text) {
  ImGui::Bullet();
  ImGui::TextUnformatted(text);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_LabelText(const char* label, const char* text) {
  ImGui::LabelText(label, "%s", text);
}

// --- Controls ---

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_Button(const char* label, float w, float h) {
  return ImGui::Button(label, ImVec2(w, h)) ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_SmallButton(const char* label) {
  return ImGui::SmallButton(label) ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_Checkbox(const char* label, int* v) {
  bool val = v ? (*v != 0) : false;
  bool result = ImGui::Checkbox(label, &val);
  if (v) *v = val ? 1 : 0;
  return result ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_RadioButton(const char* label, int active) {
  return ImGui::RadioButton(label, active != 0) ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_SliderFloat(const char* label, float* v, float v_min, float v_max, const char* format) {
  return ImGui::SliderFloat(label, v, v_min, v_max, format ? format : "%.3f") ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_SliderInt(const char* label, int* v, int v_min, int v_max, const char* format) {
  return ImGui::SliderInt(label, v, v_min, v_max, format ? format : "%d") ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_SliderFloat2(const char* label, float* v, float v_min, float v_max, const char* format) {
  return ImGui::SliderFloat2(label, v, v_min, v_max, format ? format : "%.3f") ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_SliderFloat3(const char* label, float* v, float v_min, float v_max, const char* format) {
  return ImGui::SliderFloat3(label, v, v_min, v_max, format ? format : "%.3f") ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_DragFloat(const char* label, float* v, float v_speed, float v_min, float v_max, const char* format) {
  return ImGui::DragFloat(label, v, v_speed, v_min, v_max, format ? format : "%.3f") ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_DragInt(const char* label, int* v, float v_speed, int v_min, int v_max, const char* format) {
  return ImGui::DragInt(label, v, v_speed, v_min, v_max, format ? format : "%d") ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_InputFloat(const char* label, float* v, float step, float step_fast, const char* format) {
  return ImGui::InputFloat(label, v, step, step_fast, format ? format : "%.3f") ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_InputInt(const char* label, int* v, int step, int step_fast) {
  return ImGui::InputInt(label, v, step, step_fast) ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_InputText(const char* label, char* buf, uint32_t buf_size, int flags) {
  return ImGui::InputText(label, buf, (size_t)buf_size, (ImGuiInputTextFlags)flags) ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_ColorEdit3(const char* label, float* col) {
  return ImGui::ColorEdit3(label, col) ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_ColorEdit4(const char* label, float* col, int flags) {
  return ImGui::ColorEdit4(label, col, (ImGuiColorEditFlags)flags) ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_ColorPicker3(const char* label, float* col) {
  return ImGui::ColorPicker3(label, col) ? 1 : 0;
}

// --- Combo ---

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_BeginCombo(const char* label, const char* preview_value, int flags) {
  return ImGui::BeginCombo(label, preview_value, (ImGuiComboFlags)flags) ? 1 : 0;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_EndCombo() {
  ImGui::EndCombo();
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_Selectable(const char* label, int selected, int flags, float w, float h) {
  return ImGui::Selectable(label, selected != 0, (ImGuiSelectableFlags)flags, ImVec2(w, h)) ? 1 : 0;
}

// --- Trees ---

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_CollapsingHeader(const char* label, int flags) {
  return ImGui::CollapsingHeader(label, (ImGuiTreeNodeFlags)flags) ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_TreeNode(const char* label) {
  return ImGui::TreeNode(label) ? 1 : 0;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_TreePop() {
  ImGui::TreePop();
}

// --- Tabs ---

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_BeginTabBar(const char* str_id, int flags) {
  return ImGui::BeginTabBar(str_id, (ImGuiTabBarFlags)flags) ? 1 : 0;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_EndTabBar() {
  ImGui::EndTabBar();
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_BeginTabItem(const char* label, int* p_open, int flags) {
  bool open = p_open ? (*p_open != 0) : true;
  bool result = ImGui::BeginTabItem(label, p_open ? &open : nullptr, (ImGuiTabItemFlags)flags);
  if (p_open) *p_open = open ? 1 : 0;
  return result ? 1 : 0;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_EndTabItem() {
  ImGui::EndTabItem();
}

// --- Tables ---

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_BeginTable(const char* str_id, int column, int flags, float outer_w, float outer_h, float inner_width) {
  return ImGui::BeginTable(str_id, column, (ImGuiTableFlags)flags, ImVec2(outer_w, outer_h), inner_width) ? 1 : 0;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_EndTable() {
  ImGui::EndTable();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_TableNextRow(int row_flags, float min_row_height) {
  ImGui::TableNextRow((ImGuiTableRowFlags)row_flags, min_row_height);
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_TableNextColumn() {
  return ImGui::TableNextColumn() ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_TableSetColumnIndex(int column_n) {
  return ImGui::TableSetColumnIndex(column_n) ? 1 : 0;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_TableSetupColumn(const char* label, int flags, float init_width_or_weight, uint32_t user_id) {
  ImGui::TableSetupColumn(label, (ImGuiTableColumnFlags)flags, init_width_or_weight, (ImGuiID)user_id);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_TableHeadersRow() {
  ImGui::TableHeadersRow();
}

// --- Tooltips ---

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_BeginTooltip() {
  ImGui::BeginTooltip();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_EndTooltip() {
  ImGui::EndTooltip();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_SetTooltip(const char* text) {
  ImGui::SetTooltip("%s", text);
}

// --- Misc ---

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_ProgressBar(float fraction, float w, float h, const char* overlay) {
  ImGui::ProgressBar(fraction, ImVec2(w, h), overlay);
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_IsItemHovered(int flags) {
  return ImGui::IsItemHovered((ImGuiHoveredFlags)flags) ? 1 : 0;
}

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_IsItemClicked(int mouse_button) {
  return ImGui::IsItemClicked((ImGuiMouseButton)mouse_button) ? 1 : 0;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_SetItemDefaultFocus() {
  ImGui::SetItemDefaultFocus();
}

// --- Style ---

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PushStyleColor(int idx, float r, float g, float b, float a) {
  ImGui::PushStyleColor((ImGuiCol)idx, ImVec4(r, g, b, a));
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PopStyleColor(int count) {
  ImGui::PopStyleColor(count);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PushStyleVar_Float(int idx, float val) {
  ImGui::PushStyleVar((ImGuiStyleVar)idx, val);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PushStyleVar_Vec2(int idx, float x, float y) {
  ImGui::PushStyleVar((ImGuiStyleVar)idx, ImVec2(x, y));
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PopStyleVar(int count) {
  ImGui::PopStyleVar(count);
}

// --- Plotting (ImPlot) ---

RIMGUI_EXPORT int RIMGUI_CALL remixapi_imgui_PlotBeginPlot(const char* title_id, float w, float h, int flags) {
  return ImPlot::BeginPlot(title_id, ImVec2(w, h), (ImPlotFlags)flags) ? 1 : 0;
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PlotEndPlot() {
  ImPlot::EndPlot();
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PlotPlotLine(const char* label_id, const float* values, int count) {
  ImPlot::PlotLine(label_id, values, count);
}

RIMGUI_EXPORT void RIMGUI_CALL remixapi_imgui_PlotPlotBars(const char* label_id, const float* values, int count, double bar_size) {
  ImPlot::PlotBars(label_id, values, count, bar_size);
}
