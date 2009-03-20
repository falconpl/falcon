/*
   FALCON - The Falcon Programming Language.
   FILE: rampmode.h

   Ramp mode - progressive GC limits adjustment algoritmhs
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 18 Mar 2009 19:55:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_RAMP_MODE_H
#define FALCON_RAMP_MODE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/basealloc.h>

#include <stdlib.h>

namespace Falcon {

/** Ramp-up GC parameters base class.
   The subclasses of this virtual class contain a configurable algorithm used by the
   Memory Pool to update its status after a succesful garbage collection.
*/
class FALCON_DYN_CLASS RampMode: public BaseAlloc
{
protected:
   size_t m_normal;
   size_t m_active;

public:
   RampMode() {}
   virtual ~RampMode();

   /** Called before starting a scan loop.
      No need for parameters as mempool, memory sizes and statistics are
      globally available.
   */
   virtual void onScanInit()=0;

   /** Called when the scan is complete and there is the need for a new calculation.
      No need for parameters as mempool, memory sizes and statistics are
      globally available.
   */
   virtual void onScanComplete()=0;

   /** Called when first set. */
   virtual void reset();

   /** Returns the lastly calculated memory level for the normal status. */
   size_t normalLevel() const { return m_normal; }

   /** Returns the lastly calculated memory level for the active status. */
   size_t activeLevel() const { return m_active; }
};


/** Enforces a strict inspection policy.
   The warning active level is set to the quantity of
   memory used after the last collection loop.
*/
class FALCON_DYN_CLASS RampStrict: public RampMode
{
public:
   RampStrict():
      RampMode()
   {}

   virtual ~RampStrict();
   virtual void onScanInit();
   virtual void onScanComplete();
};

#define RAMP_MODE_STRICT_ID   0


class FALCON_DYN_CLASS RampLoose: public RampMode
{
public:
   RampLoose():
      RampMode()
   {}

   virtual ~RampLoose();
   virtual void onScanInit();
   virtual void onScanComplete();
};

#define RAMP_MODE_LOOSE_ID   1

class FALCON_DYN_CLASS RampSmooth: public RampMode
{
   size_t m_pNormal;
   size_t m_pActive;
   numeric m_factor;

public:
   RampSmooth( numeric factor );
   virtual ~RampSmooth();

   virtual void reset();
   virtual void onScanInit();
   virtual void onScanComplete();
};

#define RAMP_MODE_SMOOTH_SLOW_ID   2
#define RAMP_MODE_SMOOTH_FAST_ID   3

#define RAMP_MODE_COUNT              4
#define RAMP_MODE_OFF               -1
#define DEFAULT_RAMP_MODE           RAMP_MODE_SMOOTH_SLOW_ID
}

#endif

/* end of rampmode.h */
