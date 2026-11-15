#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace spu::lifecycle {
inline uint64_t Now() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline std::string Quote(const std::string& value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) out << "?";
    else out << c;
  }
  out << '"'; return out.str();
}
struct Event {
  uint64_t id = 0, parent = 0, begin = 0, end = 0;
  std::optional<uint64_t> units;
  std::string scope, category, phase, outcome = "complete", counter_key;
  std::string request_ref, material_epoch;
  std::string measurement_status = "complete";
  std::optional<uint64_t> before, after;
};
struct State {
  std::mutex lock;
  std::atomic<int> enabled{0}; 
  std::atomic<uint64_t> coalesced_raw_chunks{0}, coalesced_cot_dispatches{0};
  std::string identity = "{}", phase = "closed", kind;
  std::string request_ref, material_epoch;
  std::set<std::string> requests;
  int rank = -1;
  uint64_t next = 1, active = 0, dropped = 0;
  std::vector<Event> events;
};
inline State& Global() { static State value; return value; }
inline std::vector<uint64_t>& Stack() { static thread_local std::vector<uint64_t> value; return value; }
inline bool Profiling(const std::string& kind) {
  return kind == "profiling" || kind == "diagnostic" || kind == "validation";
}
inline bool Transition(const std::string& from, const std::string& to) {
  if (to == "failed") return from != "closed";
  if (from == "setup") return to == "preprocessing" || to == "online" || to == "closed";
  if (from == "preprocessing") return to == "online" || to == "closed";
  if (from == "online") return to == "retirement";
  if (from == "retirement" || from == "failed") return to == "closed";
  return false;
}
inline void Begin(const std::string& identity, int rank, const std::string& kind) {
  auto& g = Global(); std::lock_guard<std::mutex> guard(g.lock);
  if (g.active || g.phase != "closed") throw std::runtime_error("previous observer session is live");
  if (rank < 0 || rank > 1 || identity.empty()) throw std::invalid_argument("public lifecycle identity required");
  g.identity = identity; g.rank = rank; g.kind = kind; g.phase = "setup";
  g.next = 1; g.dropped = 0; g.events.clear(); Stack().clear(); g.enabled.store(1);
  g.coalesced_raw_chunks.store(0); g.coalesced_cot_dispatches.store(0);
  g.requests.clear(); g.request_ref.clear(); g.material_epoch.clear();
}
inline void MarkRequest(const std::string& request_ref, const std::string& material_epoch) {
  auto& g = Global(); std::lock_guard<std::mutex> guard(g.lock);
  if (g.active || (g.phase != "setup" && g.phase != "preprocessing" && g.phase != "retirement") ||
      request_ref.empty() || material_epoch.empty() || g.requests.count(request_ref))
    throw std::runtime_error("invalid or reused lifecycle request binding");
  g.requests.insert(request_ref); g.request_ref = request_ref; g.material_epoch = material_epoch;
  g.phase = "preprocessing"; g.enabled.store(1);
}
inline void MarkPhase(const std::string& phase) {
  auto& g = Global(); std::lock_guard<std::mutex> guard(g.lock);
  if (g.active || !Transition(g.phase, phase)) throw std::runtime_error("invalid lifecycle phase transition");
  g.phase = phase;
  g.enabled.store(phase == "closed" || phase == "failed" ? 0 :
                  phase == "online" ? (Profiling(g.kind) ? 2 : 0) : 1);
}
class Scope {
 public:
  using Counter = std::function<uint64_t()>;
  Scope(std::string_view name, std::string_view category, Counter counter = {},
        std::string_view counter_key = "not-observed") : counter_(std::move(counter)) {
    auto& g = Global();
    if (!g.enabled.load(std::memory_order_relaxed)) return;

    if (name == "COT.raw-generate") { ++g.coalesced_raw_chunks; return; }
    if (name == "COT.actual-consumption-or-generation") { ++g.coalesced_cot_dispatches; return; }
    {
      std::lock_guard<std::mutex> guard(g.lock);
      if (!g.enabled.load(std::memory_order_relaxed)) return;
      active_ = true; ++g.active; event_.id = g.next++;
      event_.parent = Stack().empty() ? 0 : Stack().back();
      event_.scope = std::string(name); event_.category = std::string(category);
      event_.phase = g.phase; event_.counter_key = std::string(counter_key);
      event_.request_ref = g.request_ref; event_.material_epoch = g.material_epoch;
      Stack().push_back(event_.id);
    }
    exceptions_ = std::uncaught_exceptions();
    event_.before = ReadCounter(); event_.begin = Now();
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() noexcept { Stop(); }
  void Stop() noexcept {
    if (!active_) return;
    active_ = false;
    event_.end = Now(); event_.after = ReadCounter();
    if (std::uncaught_exceptions() > exceptions_) event_.outcome = "failed";
    if (event_.before && event_.after && *event_.after < *event_.before)
      event_.measurement_status = "counter-reset";
    try {
      auto& g = Global(); std::lock_guard<std::mutex> guard(g.lock);
      if (!Stack().empty() && Stack().back() == event_.id) Stack().pop_back();
      else event_.measurement_status = "stack-mismatch";
      if (g.events.size() < 200000) g.events.push_back(std::move(event_));
      else ++g.dropped;
      --g.active;
    } catch (...) {

      auto& g = Global(); std::lock_guard<std::mutex> guard(g.lock);
      ++g.dropped; if (g.active) --g.active;
    }
  }
  void Outcome(const std::string& outcome) { if (active_) event_.outcome = outcome; }
  void Units(uint64_t units) { if (active_) event_.units = units; }
 private:
  std::optional<uint64_t> ReadCounter() noexcept {
    if (!counter_) return std::nullopt;
    try { return counter_(); }
    catch (...) { event_.measurement_status = "counter-unavailable"; return std::nullopt; }
  }
  Counter counter_;
  Event event_;
  bool active_ = false;
  int exceptions_ = 0;
};
inline std::string Snapshot() {
  auto& g = Global(); std::lock_guard<std::mutex> guard(g.lock);
  if (g.active) throw std::runtime_error("cannot export unfinished generator work");
  std::ostringstream out;
  out << "{\"schema\":\"generator-lifecycle-v1\",\"identity\":" << g.identity
      << ",\"rank\":" << g.rank << ",\"phase\":" << Quote(g.phase)
      << ",\"kind\":" << Quote(g.kind) << ",\"dropped_events\":" << g.dropped
      << ",\"coalesced_raw_chunks\":" << g.coalesced_raw_chunks.load()
      << ",\"coalesced_cot_dispatches\":" << g.coalesced_cot_dispatches.load()
      << ",\"events\":[";
  bool first = true;
  for (const auto& e : g.events) {
    if (!first) out << ',';
    first = false;
    out << "{\"event_id\":" << e.id << ",\"parent_event\":" << e.parent
        << ",\"scope\":" << Quote(e.scope) << ",\"category\":" << Quote(e.category)
        << ",\"original_phase\":" << Quote(e.phase) << ",\"begin_ns\":" << e.begin
        << ",\"end_ns\":" << e.end << ",\"local_elapsed_ns\":" << e.end - e.begin
        << ",\"outcome\":" << Quote(e.outcome)
        << ",\"request_ref\":" << Quote(e.request_ref)
        << ",\"material_epoch\":" << Quote(e.material_epoch) << ",\"units\":";
    if (e.units) out << *e.units; else out << "null";
    out
        << ",\"counter_key\":" << Quote(e.counter_key)
        << ",\"measurement_status\":" << Quote(e.measurement_status)
        << ",\"local_sent_bytes\":";
    if (e.before && e.after && *e.after >= *e.before) out << *e.after - *e.before;
    else out << "null";
    out << ",\"time_basis\":\"local-inclusive-work-not-additive-wall\""
        << ",\"bytes_basis\":\"actual-own-sent-inclusive-or-null\""
        << ",\"time_additive\":false,\"bytes_additive\":false}";
  }
  out << "]}"; return out.str();
}
}
