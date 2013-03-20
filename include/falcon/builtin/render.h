/*
   FALCON - The Falcon Programming Language.
   FILE: render.h

   Falcon core module -- render function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 20:52:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_RENDER_H
#define	FALCON_CORE_RENDER_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
    @functions render
    @brief Renders the function as source code.
    @param item The item to be rendered.
    @optparam stream A target stream or text writer where to store the rendering.
    @return If stream is not given, a string containing the rendered function.
    @raise IoError on stream I/O error.


 */

/*#
   @method render BOM
   @brief Renders the function as source code.
   @optparam stream A target stream or text writer where to store the rendering.
   @return If stream is not given, a string containing the rendered function.
   @raise IoError on stream I/O error.
   @see render
*/

class FALCON_DYN_CLASS Render: public Function
{
public:
   Render();
   virtual ~Render();
   virtual void invoke( VMContext* vm, int32 nParams );
};

}
}

#endif	/* FALCON_CORE_RENDER_H */

/* end of render.h */
