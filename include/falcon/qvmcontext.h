/*
   FALCON - The Falcon Programming Language.
   FILE: QVMContext.h

   Falcon virtual machine -- queued VM context.
   -------------------------------------------------------------------
   Author: AlexRou

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_QVMCONTEXT_H_
#define FALCON_QVMCONTEXT_H_

#include <falcon/setup.h>
#include <falcon/vmcontext.h>
#include <falcon/mt.h>
#include <falcon/itemarray.h>

namespace Falcon
{

class String;
/**
* Queued vm context runs items in a first-in-first-out queue
* Will not terminate unless terminate() is called and will prevent the program from closing
*/
class FALCON_DYN_CLASS QVMContext: public VMContext
{
	public:
		QVMContext( Process* prc, ContextGroup* grp = 0 );
		virtual ~QVMContext();

		/** Adds a item to the queue
		* PSteps complete and error require 2 popCode()
		* If no complete PStep is given the return value is ignored
		* If no error PStep is given and a error is caught, the error message will go to the error stream
		*/
		void start( String name, Item item, int32 np = 0, Item const* params = 0, PStep* complete = 0, PStep* error = 0 );

		virtual void gcPerformMark();

		/** Shouldn't be called until terminate() */
		virtual void onComplete();

		/** Returns true when there is nothing in the queue */
		bool isSleeping()
		{
			return m_isSleeping;
		}

		/** Name of currently running item or "Sleeping" when queue is empty */
		String runningItem()
		{
			return m_running;
		}

	private:

		class QueueItem;

		typedef struct Node
		{
			QueueItem* data;
			Node* next;
		} Node;

		class PStepComplete;
		class PStepErrorGate;

		class PStepDeque;
		PStep* m_stepQueue;

		Function* m_baseFrame;

		Node* m_Queue;
		Node* m_QueueEnd;
		bool m_isSleeping;

		String m_running;
};

}

#endif /* FALCON_QVMCONTEXT_H_ */

/* end of QVMContext.h */
