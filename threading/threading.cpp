/*
   FALCON - The Falcon Programming Language.
   FILE: threading.cpp

   Multithreading support - main file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 10 Apr 2008 00:44:09 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Multithreading support - main file.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include "version.h"
#include "threading_ext.h"
#include "threading_st.h"
#include <systhread.h>

/*#
   @module feathers_threading Threading
   @brief Multithreading Support Module.

   This module provides support for real hardware-supported
   parallel execution of concurrent code under the same
   application hood; this is commonly called "Multithreading".

   By linking this module in the virtual machine through a load
   directive, or through the reflexive compiler module, the
   VM gets prepared to launch system threads and perform
   multithreading operations.

   See the related page @a threading_model for a description
   on the general aspects and design of the multithreading
   provided by this module.

   The @a threading_warnings page gives an important insight about
   overall security of multithreading applications (and
   specifically about MT security in Falcon).

   The @a threading_stop page explains about the conformance
   of this module with the VM Interruption protocol.
*/

/*#
   @page threading_model Falcon multithreading model.

   @section Forewords

   This document is by no mean an exhaustive explanation of multithreading
   in general. Concepts as "mutex", "synchronization primitive", "thread",
   and the like are given for granted. The reader should already know
   the basics of multithreading and have an idea on the topic, as this
   document just deals with the specificities of Falcon approach to
   multithreading.

   @section Basic Principles

   Falcon multithreading is amied to maximize the efficiency of the VM
   running in a multithreading context, and of the execution of the scripts
   in separate threads.

   Principles of efficient and robust multithreading can be resumed in two points:
   - Threads must run unhindered and free from synchronization with the rest of
     the application for the vast part of their life. Data exchange with other
     threads must happen rarely, and it must take a fraction of the time needed
     to perform data elaboration.
   - Synchronization must happen at the topmost layers of the logic controlling
     threads, as it is a critical operation in their life, which must be given
     maximum care and control. Burying synchronization down in the lower layers
     of code, or worse, hiding its presence through class encapsulation is to be
     avoided.

   While real world is not perfect and there can be exceptions to this rules,
   using this two simple principles as a guide it is possible to write programs
   which maximally exploit the parallel computing facilities that modern computers
   provide, eliminating the risk of incurring in multithreading programming specific
   errors, as races, deadlocks and similar.

   Scripting languages, as Falcon, perform a lot of operations in background that
   are out of the control of the script writers, and this makes them a quite hard
   landscape for multithreading. A rule that can be considered a corollary to the
   two main principles, "mutexes must be held for the shortest possible time", is
   quite hard to be respected when the simplest istruction in a scripting language
   can take many complex actions (in the order of hundreds) at machine code level.

   For this reason, Falcon threading enforces the above principles with a "pure
   agent-based threading model".

   Each Falcon thread is an agent, bound to perform non-trivial and long-lasting
   operations, which can exchange data with the other agents in the application
   through objects called "synchronization structures". Structures are relatively
   complex objects that allow safe communication and interaction between threads.

   Each agent has its own application space where it is free to perform operations
   unhindered by the intervention of any other thread or by the Falcon engine.
   Exchange of data with the rest of the application can happen only thorough
   synchronization structures. Some of this structures are quite strightforward
   to be used, and in example, it is possible to share plain memory which can
   be directly manipulated by each thread as it prefers.

   @section Multithreading implementation

   The Falcon threading module provides each agent with a new Virtual Machine
   created on the spot. Those VMs are created "empty", that is, they will
   contain only the modules that were linked by the VM that started the thread
   as it was right before starting its execution.

   VM related operations, as setting the garbage collection properties, termination
   requests, sleep requests, exceptions, memory pools and garbage collection loops
   are all local to a certain agent.

   Exchange of Falcon items, as objects, strings, vectors and so on is performed through
   serialization in memory. As each item lives and resides in a VM, each item must stay
   consistent to the VM that created it. Providing other agents the ability to change
   agent data would require synchronization at deep level in very frequent spots,
   and would rapidly make a script multithreading application run almost a thread
   at a time, instead of being fully parallel.

   However, the fact that items need to be serialized to be shared among agents doesn't
   necessarily means that they can't share memory. Falcon items are shells, representation
   of inner data which is provided by system-level code. While the items, which carry those
   data, must be copied between agents, the inner core which interacts with the system
   can just be shared and provided with proper synchronization at system level.

   In example, it is possible to share Falcon streams between agents. Each agent will have
   its own copy of the stream object, with a local view of stream data as i.e. the number
   of bytes written in the last operation, or the position in a file, but the underlying
   system resource will be shared and concurrency will be regulated by system calls.

   Embedding applications or other modules willing to work in multithreading can adopt the
   same strategy. Falcon gives the embedding applications and the module the ability to
   carry their data in a Falcon object, and receive relevant callbacks when a script
   needs to interact with that data. When an object is cloned for serialization, the embedding
   application will be notified, and it may perform proper action to prepare the data
   to be shared among threads. It may be as simple as a MT safe reference counting, or it may
   require more sophisticate operations; in example, sharing a Falcon Stream requires the inner
   layer of code to call a dup() system request to ask the operating system to create a
   duplicated file resource.

   Once shared between threads in this way, application or module data must take actions
   to ensure proper synchronization. That is, property access and method call must be
   prepared to be called by different threads concurrently.

   In this sense, it may be said that Falcon doesn't provide a "memory model", but allows
   each object to provide its own. While this may be thought as "confusing" for the script
   writers, once that the overall rules of the system (nothing is shared but...) plus the
   specific rules of the shared objects actually used by the script
   (... except this thing, when you do so) are known and followed, the overall complexity
   of a MT application built following this approach is by no mean higher than the complexity
   of a MT application built on a layer with a consistent and unique memory model.

   Overall complexity of a MT application depends on the data flow, and primarily on the
   synchronization logic and on the interactions between threads. There is nothing preventing
   an application with local, object specific memory models to be actually less complex than
   one with a burned-in memory model. The constraints given by each synchronization structure,
   which may have different visibility and sharing rules, ensure that a simple set of rules
   are valid locally, while the rest of the program is simply "safe and local". This actually
   works towards simplification and legibility of MT code.

   This may requires deep-level
   synchronization, which seems to contrast with the overall principles enunciated at the
   beginning of this section; but as long as this synchronization is kept minimal
   (i.e. just to
   ensure visibility of shared properties), or as long as the synchronization rules are
   available, known and controllable by the topmost level, the overall agent-based model
   is not broken.

   @section Synchronization structures

   Falcon agent-based model leverages on the concept of non-primitive structures used to
   synchronize and coordinate threads. Each structure has a different working principle which
   is explained in its detailed description, but they all share the concept of @b wait, @b acquisition
   and @b release.

   An agent can @b wait for one or more structures to be ready for acquisition. When this happens,
   the agent is notified and it acquires atomically one of the structures it was waiting for.

   Once perfomred the work for which the structure was needed, the agent must @b release the
   structure, and is then free to wait again or terminate.

   Acquisition and release are not necessarily bound to exclusive access. There are structures
   that may be acquired by many threads, or others that can only be acquired (their @b release is
   an void operation). The concept of @b acquisition is rather a semantic equivalent to "allowance",
   "permission" to perform certain operations bound with the structure.

   More details are explained in the description of each structure.
*/

/*#
   @page threading_warnings Multithreading safety

   @section Multithreading can break things

   Falcon won't try to recover from multithreading errors done by the scripts. This means that
   an ill designed script can deadlock, mess up embedding application data and break in the most
   funny ways. If a script deadlocks, you won't be able to destroy its VM from an application
   and recover its resources; when a multithreaing thing (real full system level MT) in your
   application goes wild, your application is done for good. This even if the thing is a
   script encapsulated in a scripting engine.

   Also, this is not just unavoidable (don't believe in who says it can be avoided), but also
   desirable. When a MT problem happens, this leaves (or may have leaved) the application
   in an unconsistent state, an thus it may create errors in other parts of the application,
   or produce inconsistent results. If the application forcefully clears the inconsistent
   status of the MT sublayer, i.e. by forcefully releasing a mutex in deadlocked state,
   this has an high (and unmanageable) change to cause an unforecastable and possibly
   undetectable error somewhere else.

   As an example, think of a generic algorithm that locks a mutex, changes the name
   and the surname in a record and unlocks the mutex. Just, let's suppose we are able
   to change the name, but the surname is too long and the update method raises
   an exception. With automatic unlock, we'd have something like the following:

   @code
      // this is pseudo code; we're using a java/c# like syntax
      void updateRecord( String name, String surname )
                         throws something
      {
         AutoLock locker( this.mutex );
         this.setName( name );
         this.setSurname( surname ); // throws the exception.
      }
   @endcode

   In the above example we ipotized an AutoLock class that locks a mutex, then unlocks it
   at stack unroll.

   Now, that guard ensure that we release the mutex at function exit, no matter how it exits.
   Pitifully, it can't grant data to be consistent. In example, we may have set our record
   to a new name, "Adam aVeryVeryLongStrangerSurname", and get the exception on the surname.
   Adam will be stored, and we'll have something as an unexistent "Adam Smith" in our database.
   We broken our data, and we may know that much later, in example when Mr. aVeryVeryLong...etc,
   our boss, fires us because we didn't send him his montly wage payroll,
   but "yo, ye mutex is unlocked"!

   Of course, we could have gotten the exception and fix things; but while we
   were fixing things we may also have properly released the mutex when it
   was consistent to do it.

   But the point is another. If we made such an error as forgetting to take care
   of the dirty record above, then the most immediate and effective diagniostic is that of
   being locked on it. Compare the above with this code:

   @code
   // this is pseudo code; we're using a java/c# like syntax
   void updateRecord( String name, String surname )
                        throws something
   {
      this.mutex.lock();
      this.setName( name );
      this.setSurname( surname ); // throws the exception.
      this.mutex.unlock();
   }
   @endcode

   The logic error is the same as above. We have broken the name, but now the MT
   layer is in inconsistent state, and the next lock to the mutex will cause
   the application to deadlock. Provided mutexes are used rationally, this
   means that we'll have a clear indication of what appened in the early stages
   of development.

   Everyone forgets to unlock mutexes, at spots, and when it happens it hurts,
   because you feel like an idiot. Feeling like an idiot is the worst thing
   for a programmer. This is very probably the reason why such automatic one-line
   savers have been invented. However, clearing the MT layer status without
   being able to be sure that the application built on the MT layer is clear too
   is a very dangerous operation.

   As such, Falcon doesn't provide it. If you acquire a synchronized structure
   you have to explicitly release it, or it won't be available for others to
   be acquired again. This is because it would be impossible, for Falcon, to
   know which part of the application state has been left inconsistent together
   with the synchronization structure. If Falcon release a shared resource
   in behalf of a failing thread, this may cause other working threads to
   use the shared data that was half worked and incomplete, and this may
   lead to disasters that are far worse than deadlock.

   The correct approach to MT failures is not that to hide them or to
   automatically "fix" (hide) them, it is that to cure the program generating
   them so that they don't happen anymore.

   @section Multithreading can break things (again)

   We said that that
   an ill designed script can deadlock, mess up embedding application data and break in the most
   funny ways.

   This means that embedding applications must be carefull in allowing their scripts to use threads.
   In case of security sensible applications, the threading module should be pre-loaded by
   the application and provided only to those scripts that have an administrative or equivalent
   security level. Only trusted scripts should be allowed to run MT in a production environment.
*/

/*#
   @page threading_stop Threading module and VM Interruption protocol

   Interruptible operations will raise an InterruptedError if they receive
   an asynchronous interruption request from another thread while ingaged
   in a lengthy wait.

   The Thread.wait method conforms to the VM Interruption protocol,
   along with other operations that are declared by the core module or
   by other extensions. Other interruptible operations are, currently:
   - sleep function.
   - Stream.readAvailable
   - Stream.writeAvailable
   - Socket.readAvailable (at the moment, only on non-windows platforms)
   - Socket.writeAvailable (at the moment, only on non-windows platforms)

   We are working to extend the protocol to other VM-level and sustem level
   operations.

   Other than coming from embedding applications, interruption requests
   can be generated by the Thread.stop method, that will send an asynchronous
   interruption request to the target thread.
*/

/*#
   @beginmodule feathers_threading
*/

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   Falcon::Module *self = new Falcon::Module();
   self->name( "threading" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );
   //====================================
   // Message setting
   #include "threading_st.h"

   //=================================================================
   // Threading class -- enumeration for threading functions.
   //
   Falcon::Symbol *c_threading = self->addClass( "Threading" );
   self->addClassMethod( c_threading, "wait", Falcon::Ext::Threading_wait ).asSymbol()->
      addParam("waitTime");
   self->addClassMethod( c_threading, "vwait", Falcon::Ext::Threading_vwait ).asSymbol()->
      addParam("structArray")->addParam("waitTime");
   self->addClassMethod( c_threading, "getCurrentID", Falcon::Ext::Threading_getCurrentID );
   self->addClassMethod( c_threading, "getCurrent", Falcon::Ext::Threading_getCurrent );
   self->addClassMethod( c_threading, "sameThread", Falcon::Ext::Threading_sameThread ).asSymbol()->
      addParam("thread");
   self->addClassMethod( c_threading, "start", Falcon::Ext::Threading_start ).asSymbol()->
      addParam("callable");

   //=================================================================
   // Waitable class.
   //
   Falcon::Symbol *c_waitable = self->addClass( "Waitable" );
   c_waitable->exported( false );
   self->addClassMethod( c_waitable, "release", Falcon::Ext::Waitable_release );

   //=================================================================
   // Thread class.
   //
   Falcon::Symbol *c_thread = self->addClass( "Thread", Falcon::Ext::Thread_init );
   c_thread->getClassDef()->addInheritance( new Falcon::InheritDef( c_waitable ) );
   c_thread->setWKS( true );
   self->addClassMethod( c_thread, "start", Falcon::Ext::Thread_start );
   self->addClassMethod( c_thread, "stop", Falcon::Ext::Thread_stop );
   self->addClassMethod( c_thread, "detach", Falcon::Ext::Thread_detach );
   self->addClassMethod( c_thread, "wait", Falcon::Ext::Thread_wait ).asSymbol()->
      addParam("waitTime");
   self->addClassMethod( c_thread, "vwait", Falcon::Ext::Thread_vwait ).asSymbol()->
      addParam("structArray")->addParam("waitTime");
   self->addClassMethod( c_thread, "getError", Falcon::Ext::Thread_getError );
   self->addClassMethod( c_thread, "getReturn", Falcon::Ext::Thread_getReturn );
   self->addClassMethod( c_thread, "hadError", Falcon::Ext::Thread_hadError );
   self->addClassMethod( c_thread, "getThreadID", Falcon::Ext::Thread_getThreadID );
   self->addClassMethod( c_thread, "sameThread", Falcon::Ext::Thread_sameThread ).asSymbol()->
      addParam("otherThread");
   self->addClassMethod( c_thread, "terminated", Falcon::Ext::Thread_terminated );
   self->addClassMethod( c_thread, "detached", Falcon::Ext::Thread_detached );
   self->addClassMethod( c_thread, "join", Falcon::Ext::Thread_join );
   self->addClassMethod( c_thread, "getSystemId", Falcon::Ext::Thread_getSystemID );
   self->addClassMethod( c_thread, "setName", Falcon::Ext::Thread_setName ).asSymbol()->
      addParam("name");
   self->addClassMethod( c_thread, "getName", Falcon::Ext::Thread_getName );
   self->addClassMethod( c_thread, "toString", Falcon::Ext::Thread_toString );
   self->addClassProperty( c_thread, "run" );

   //=================================================================
   // Grant class.
   //
   Falcon::Symbol *c_grant = self->addClass( "Grant", Falcon::Ext::Grant_init );
   c_grant->getClassDef()->addInheritance( new Falcon::InheritDef( c_waitable ) );

   //=================================================================
   // Barrier class.
   //
   Falcon::Symbol *c_barrier = self->addClass( "Barrier", Falcon::Ext::Barrier_init );
   c_barrier->getClassDef()->addInheritance( new Falcon::InheritDef( c_waitable ) );
   self->addClassMethod( c_barrier, "open", Falcon::Ext::Barrier_open );
   self->addClassMethod( c_barrier, "close", Falcon::Ext::Barrier_close );

   //=================================================================
   // Event class.
   //
   Falcon::Symbol *c_event = self->addClass( "Event", Falcon::Ext::Event_init );
   c_event->getClassDef()->addInheritance( new Falcon::InheritDef( c_waitable ) );
   self->addClassMethod( c_event, "set", Falcon::Ext::Event_set );
   self->addClassMethod( c_event, "reset", Falcon::Ext::Event_reset );

   //=================================================================
   // Counter class.
   //
   Falcon::Symbol *c_synccount = self->addClass( "SyncCounter", Falcon::Ext::SyncCounter_init );
   c_synccount->getClassDef()->addInheritance( new Falcon::InheritDef( c_waitable ) );
   self->addClassMethod( c_synccount, "post", Falcon::Ext::SyncCounter_post ).asSymbol()->
      addParam("count");

   //=================================================================
   // SyncQueue class.
   //
   Falcon::Symbol *c_synq = self->addClass( "SyncQueue", Falcon::Ext::SyncQueue_init );
   c_synq->getClassDef()->addInheritance( new Falcon::InheritDef( c_waitable ) );
   self->addClassMethod( c_synq, "push", Falcon::Ext::SyncQueue_push ).asSymbol()->
      addParam("item");
   self->addClassMethod( c_synq, "pushFront", Falcon::Ext::SyncQueue_pushFront ).asSymbol()->
      addParam("item");
   self->addClassMethod( c_synq, "pop", Falcon::Ext::SyncQueue_pop );
   self->addClassMethod( c_synq, "popFront", Falcon::Ext::SyncQueue_popFront );
   self->addClassMethod( c_synq, "empty", Falcon::Ext::SyncQueue_empty );
   self->addClassMethod( c_synq, "size", Falcon::Ext::SyncQueue_size );

   //============================================================
   // Thread Error class
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *thread_cls = self->addClass( "ThreadError", Falcon::Ext::ThreadError_init );
   thread_cls->setWKS( true );
   thread_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   //============================================================
   // Join Error class
   Falcon::Symbol *joinerr_cls = self->addClass( "JoinError", Falcon::Ext::JoinError_init );
   joinerr_cls->setWKS( true );
   joinerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   //==================================================================
   // Service feature.
   //
   //self->publishService( &the_service );

   // THIS MODULE SHALL NOT BE RELEASED for now.
   self->incref();

   return self;
}

/* end of threading.cpp */

