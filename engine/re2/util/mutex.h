// Copyright 2007 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * FALCON PROGRAMMING LANGUAGE
 * The RE2 mutex has been changed into the Falcon
 * mutex.
 *
 * You should assume the locks are *not* re-entrant.
 */

#ifndef RE2_UTIL_MUTEX_H_
#define RE2_UTIL_MUTEX_H_

#include <falcon/mt.h>

namespace re2 {

class Mutex: Falcon::Mutex {
 public:
  // Create a Mutex that is not held by anybody.
  inline Mutex() {}

  // Destructor
  inline ~Mutex() {};

  inline void Lock()   {lock();}
  inline void Unlock()   {unlock();}
  inline void TryLock()   {trylock();}

  inline void ReaderLock()   {lock();}
  inline void ReaderUnlock() {unlock();}
  inline void WriterLock() { lock(); }     // Acquire an exclusive lock
  inline void WriterUnlock() { unlock(); } // Release a lock from WriterLock()
  inline void AssertHeld() { }

 private:
  // Catch the error of writing Mutex when intending MutexLock.
  Mutex(Mutex *) {}
  // Disallow "evil" constructors
  Mutex(const Mutex&);
  void operator=(const Mutex&);
};

// --------------------------------------------------------------------------
// Some helper classes

// MutexLock(mu) acquires mu when constructed and releases it when destroyed.
class MutexLock {
 public:
  explicit MutexLock(Mutex *mu) : mu_(mu) { mu_->Lock(); }
  ~MutexLock() { mu_->Unlock(); }
 private:
  Mutex * const mu_;
  // Disallow "evil" constructors
  MutexLock(const MutexLock&);
  void operator=(const MutexLock&);
};

// ReaderMutexLock and WriterMutexLock do the same, for rwlocks
class ReaderMutexLock {
 public:
  explicit ReaderMutexLock(Mutex *mu) : mu_(mu) { mu_->ReaderLock(); }
  ~ReaderMutexLock() { mu_->ReaderUnlock(); }
 private:
  Mutex * const mu_;
  // Disallow "evil" constructors
  ReaderMutexLock(const ReaderMutexLock&);
  void operator=(const ReaderMutexLock&);
};

class WriterMutexLock {
 public:
  explicit WriterMutexLock(Mutex *mu) : mu_(mu) { mu_->WriterLock(); }
  ~WriterMutexLock() { mu_->WriterUnlock(); }
 private:
  Mutex * const mu_;
  // Disallow "evil" constructors
  WriterMutexLock(const WriterMutexLock&);
  void operator=(const WriterMutexLock&);
};

// Catch bug where variable name is omitted, e.g. MutexLock (&mu);
#define MutexLock(x) COMPILE_ASSERT(0, mutex_lock_decl_missing_var_name)
#define ReaderMutexLock(x) COMPILE_ASSERT(0, rmutex_lock_decl_missing_var_name)
#define WriterMutexLock(x) COMPILE_ASSERT(0, wmutex_lock_decl_missing_var_name)

}  // namespace re2

#endif  /* #define RE2_UTIL_MUTEX_H_ */
