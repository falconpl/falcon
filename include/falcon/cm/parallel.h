/*
   FALCON - The Falcon Programming Language.
   FILE: parallel.h

   Falcon core module -- Interface to Parallel class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 29 Nov 2012 13:52:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARALLEL_H_
#define _FALCON_PARALLEL_H_

#include <falcon/setup.h>
#include <falcon/fassert.h>
#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*# Parallel paradigm entry point.

This class manages parallel processing performed on one or more parallel computations.

The actual parallelization on physical computational resources is partially controlled
by the engine and partially by this class, and should considered to be mostly transparent
to the final Falcon application; however, if
it is desired to use a coroutining parallel model (apparent lightweight parallelism),
this class can be configured to run just one of the parallel executions at a time, and
then all the parallelism will be performed through timeslicing on a single computational
resource.

@section usage Usage

The Parallel class is first created with all the parallel computations that must be performed,
that are a set of callable items (functions, arrays starting with a function, object functors
etc.). Each passed callable will be the main function of a @i parallel @i agent, which is the
logical computational managed by the Parallel class.

After that, it is possible to configure the execution that will be later performed, for instance
subscribing all the agents to one or more message boxes.

When the parallel computation is fully created, it is launched with either @a Parallel.launch or
@a Parallel.launchWithResults. Both the methods are fully blocking; the invoking agent will be
suspended until all the agents have completed their computation. External logic controlling
the termination of launched agents must either be in one or more of the started agents, or come
from another parallelized agent somewhere else.

The method @a Parallel.launchWithResults is identical to @a Parallel.launch, but the first one
will also return an array containing all the return values of the main functions of the parallelized
agent.

The following is a minimal example.

@code

function calc( a, b )
   return a+b
end

x = Parallel( .[calc 1 2], .[calc 3 4] )
result = x.launchWithResults()

> result          // [3, 7]

@endcode

@section error Error management

If any of the agents generate an uncaught error, the whole parallel
execution is terminated, and an @a AgentError is thrown in the caller
agent.

The @a AgentError class contains extra context information about
which agent caused the error, and obviously the error thrown.

In case multiple errors are thrown after the first error is
detected but before the whole of the agent group is terminated,
they are silently discarded. The rationale is that managing them
would be meaningless until the first detected cause is cured.

@section typical Typical usage pattern

A well formed, agent-based parallel execution is usually configured so to:

- Use functor objects to have a greater control over parallel execution context
- Have a control agent deciding when it's time to conclude the computation.
- Use a barrier or a message to communicate across agents.
- Manage the AgentError that could be thrown.

The following is a typical well structured parallel execution agent template
framework. It's a simple single producer, multiple consumer pattern where
a producer generates a string to be handled once per second, and then a consumer
(at random) extracts it, until a control agent opens a barrier that declares
an arbitrary to be expired. In the example, the control agent is just initiated
with a function for simplicity.

@code
class Consumer( name, killer, resource )
   name = name
   killer = killer
   resource = resource

   function __call()
      while Parallel.wait( self.resource, self.killer ) != self.killer
         > "Agent ", self.name, " received: ", self.resource.pop()
      end
   end
end


class Producer( name, killer, resource )
   name = name
   killer = killer
   resource = resource

   function __call()
      i = 0
      while Parallel.timedWait( 1, self.killer ) != self.killer
         self.resource.push("More work... " + i)
      end
   end
end

k = Barrier()
r = SyncQueue()
p = Parallel(
   Producer( "producer", k, r ),
   Consumer( "consumer1", k, r ),
   Consumer( "consumer2", k, r ),
   // we'll just use  a function here for simplicity
   {=>sleep(10); k.open();}
   )

try
   p.launch()
   > "Work terminated."

catch AgentError in e
   > @"Agent $(e.agent.name)($(e.agentId)) has crashed:"
   > e.error
end

@endcode

As the @a AgentError.agent property holds the item that initiated the agent causing
an error, it will be one of our Consumer
or Producer instance (presumably, the control agent initiated with a function
won't raise any error), hence we're able to address its @b name property.
*/
class ClassParallel: public ClassUser
{
public:
   ClassParallel();
   virtual ~ClassParallel();

   /*
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const;
   */
   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void* instance, int pcount ) const;
   //virtual void op_toString( VMContext* ctx, void* self ) const;

private:

   class FALCON_DYN_CLASS PStepGetResults: public PStep
   {
   public:
      PStepGetResults();
      virtual ~PStepGetResults() {}
      static void apply_( const PStep* ps, VMContext* ctx );
   };

   PStepGetResults m_getResult;

   FALCON_DECLARE_METHOD( wait, "..." );
   FALCON_DECLARE_METHOD( timedWait, "timeout:N,..." );
   FALCON_DECLARE_METHOD( launch, "..." );
   FALCON_DECLARE_METHOD( launchWithResults, "..." );
};

}
}


#endif /* PARALLEL_H_ */

/* end of parallel.h */
