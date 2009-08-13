/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_ext.cpp

   The SDL True Type binding support module.
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Tue, 12 Aug 2009 00:06:56 +1100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL OpenGL binding support module.
*/

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include <falcon/autocstring.h>
#include <falcon/membuf.h>

#include "gl_ext.h"
#include "gl_mod.h"
#include <sdl_service.h>  // for the reset
#include <SDL.h>
#include <SDL_opengl.h>


/*# @beginmodule sdlopengl */

namespace Falcon {

static SDLService *s_service = 0;

namespace Ext {
   
   FALCON_FUNC opengl_Viewport( ::Falcon::VMachine *vm )
   {
      Item *i_x = vm->param(0);
      Item *i_y = vm->param(1);
      Item *i_width = vm->param(2);
      Item *i_height = vm->param(3);
      if ( ( i_width == 0 || ! i_width->isOrdinal() ) ||
           ( i_height == 0 || ! i_height->isOrdinal() ) ||
           ( i_x == 0 || ! i_x->isOrdinal() ) ||
           ( i_y == 0 || ! i_y->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N,N,N" ) ) ;
         return;
      }
      
      int x = (int) i_x->forceInteger();
      int y = (int) i_y->forceInteger();
      int width = (int) i_width->forceInteger();
      int height = (int) i_height->forceInteger();
      
      ::glViewport(x,y,width,height);
   }

   FALCON_FUNC sdlgl_SetAttribute( ::Falcon::VMachine *vm )
   {
      Item *i_attr = vm->param(0);
      Item *i_value = vm->param(1);
      if ( ( i_attr == 0 || !i_attr->isOrdinal() ) ||
           ( i_value == 0 || !i_value->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N" ) ) ;
         return;
      }
      SDL_GLattr attr = (SDL_GLattr)i_attr->forceInteger();
      int value = (int)i_value->forceInteger();
      int retval = ::SDL_GL_SetAttribute(attr, value);
      vm->retval(retval);

   }
   FALCON_FUNC sdlgl_GetAttribute( ::Falcon::VMachine *vm )
   {
      Item *i_attr = vm->param(0);
      Item *i_value = vm->param(1);
      if ( ( i_attr == 0 || ! i_attr->isOrdinal() ) ||
           ( i_value == 0 || ! i_value->isReference() ) || 
           ( (i_value->dereference()) == 0 || ! i_value->dereference()->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,$N" ) ) ;
         return;
      }
      SDL_GLattr attr = (SDL_GLattr)i_attr->forceInteger();
      int * value = (int *)i_value->dereference()->forceInteger();
      int retval = ::SDL_GL_GetAttribute(attr, value);
      *i_value = (int64) *value;
      vm->retval(retval);
   }

   FALCON_FUNC opengl_LoadIdentity( ::Falcon::VMachine *vm )
   {
      ::glLoadIdentity();
   }

   FALCON_FUNC opengl_MatrixMode( ::Falcon::VMachine *vm )
   {
      Item *i_matrix = vm->param(0);
      if ( (i_matrix == 0 || !i_matrix->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N" ) ) ;
         return;
      }
      GLenum matrix = (GLenum)i_matrix->forceInteger();
      ::glMatrixMode(matrix);
   }

   FALCON_FUNC opengl_ClearColor( ::Falcon::VMachine *vm )
   {
      Item *i_red = vm->param(0);
      Item *i_green = vm->param(1);
      Item *i_blue = vm->param(2);
      Item *i_alpha = vm->param(3);
      if ( ( i_red == 0 || !i_red->isOrdinal() ) ||
           ( i_green == 0 || !i_green->isOrdinal() ) ||
           ( i_blue == 0 || !i_blue->isOrdinal() ) ||
           ( i_alpha == 0 || !i_alpha->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N,N,N" ) ) ;
         return;
      }
      GLclampf red = (GLclampf) i_red->forceNumeric();
      GLclampf green = (GLclampf) i_green->forceNumeric();
      GLclampf blue = (GLclampf) i_blue->forceNumeric();
      GLclampf alpha = (GLclampf) i_alpha->forceNumeric();

      ::glClearColor(red, green, blue, alpha);
   
   }

   FALCON_FUNC opengl_ClearDepth( ::Falcon::VMachine *vm )
   {
      Item *i_depth = vm->param(0);
      if ( ( i_depth == 0 || !i_depth->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N" ) ) ;
         return;
      }

      GLclampd depth = (GLclampd) i_depth->forceNumeric();
      ::glClearDepth(depth);
   }

   FALCON_FUNC opengl_Enable( VMachine *vm )
   {
      Item *i_cap = vm->param(0);
      if ( ( i_cap == 0 || !i_cap->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N" ) ) ;
         return;
      }

      GLenum cap = (GLenum ) i_cap->forceInteger();
      ::glEnable ( cap);
   }

   FALCON_FUNC opengl_Clear( VMachine *vm )
   {
      Item *i_mask = vm->param(0);
      if ( ( i_mask == 0 || !i_mask->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N" ) ) ;
         return;
      }

      GLbitfield  mask = (GLbitfield ) i_mask->forceInteger();
      ::glClear(mask);
   }

   FALCON_FUNC opengl_Begin( VMachine *vm )
   {
      Item *i_mode = vm->param(0);
      if ( ( i_mode == 0 || !i_mode->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N" ) ) ;
         return;
      }

      GLenum  mode = (GLenum ) i_mode->forceInteger();
      ::glBegin(mode);
   }

   FALCON_FUNC opengl_End( VMachine *vm )
   {
      ::glEnd();
   }

   FALCON_FUNC opengl_SwapBuffers( VMachine *vm )
   {
      ::SDL_GL_SwapBuffers();
   }

   FALCON_FUNC opengl_Color3d( VMachine *vm )
   {
      Item *i_red = vm->param(0);
      Item *i_green = vm->param(1);
      Item *i_blue = vm->param(2);
      if ( ( i_red == 0 || !i_red->isOrdinal() ) ||
           ( i_green == 0 || !i_green->isOrdinal() ) ||
           ( i_blue == 0 || !i_blue->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N,N" ) ) ;
         return;
      }
      GLdouble red = (GLdouble) i_red->forceNumeric();
      GLdouble green = (GLdouble) i_green->forceNumeric();
      GLdouble blue = (GLdouble) i_blue->forceNumeric();
      

      ::glColor3d(red, green, blue);
   }

   FALCON_FUNC opengl_Vertex3d( VMachine *vm )
   {
      Item *i_x = vm->param(0);
      Item *i_y = vm->param(1);
      Item *i_z = vm->param(2);
      if ( ( i_x == 0 || !i_x->isOrdinal() ) ||
           ( i_y == 0 || !i_y->isOrdinal() ) ||
           ( i_z == 0 || !i_z->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N,N" ) ) ;
         return;
      }
      GLdouble x = (GLdouble) i_x->forceNumeric();
      GLdouble y = (GLdouble) i_y->forceNumeric();
      GLdouble z = (GLdouble) i_z->forceNumeric();
      

      ::glVertex3d(x, y, z);
   }

   FALCON_FUNC opengl_Translate( VMachine *vm )
   {
      Item *i_x = vm->param(0);
      Item *i_y = vm->param(1);
      Item *i_z = vm->param(2);
      if ( ( i_x == 0 || !i_x->isOrdinal() ) ||
           ( i_y == 0 || !i_y->isOrdinal() ) ||
           ( i_z == 0 || !i_z->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N,N" ) ) ;
         return;
      }
      GLdouble x = (GLdouble) i_x->forceNumeric();
      GLdouble y = (GLdouble) i_y->forceNumeric();
      GLdouble z = (GLdouble) i_z->forceNumeric();
      

      ::glTranslated(x, y, z);
   }

   FALCON_FUNC opengl_Rotate( VMachine *vm )
   {
      Item *i_angle = vm->param(0);
      Item *i_x = vm->param(1);
      Item *i_y = vm->param(2);
      Item *i_z = vm->param(3);
      if ( ( i_angle == 0 || !i_angle->isOrdinal() ) ||
           ( i_x == 0 || !i_x->isOrdinal() ) ||
           ( i_y == 0 || !i_y->isOrdinal() ) ||
           ( i_z == 0 || !i_z->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N,N,N" ) ) ;
         return;
      }
      GLdouble angle = (GLdouble) i_angle->forceNumeric();
      GLdouble x = (GLdouble) i_x->forceNumeric();
      GLdouble y = (GLdouble) i_y->forceNumeric();
      GLdouble z = (GLdouble) i_z->forceNumeric();
      
      ::glRotated(angle, x, y, z);
   }

   FALCON_FUNC opengl_GenTextures( VMachine *vm )
   {
      Item *i_num = vm->param(0);
      if ( ( i_num == 0 || !i_num->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N" ) ) ;
         return;
      }

      GLsizei  num = (GLsizei ) i_num->forceInteger();
      GLuint *textures = new GLuint[num];
      ::glGenTextures(num, textures);
      CoreArray *i_textures = new CoreArray(num);
      for (int i = 0; i < num; i++)
      {
         i_textures->append(Item(textures[i]));
      }
      delete [] textures;
      vm->retval(i_textures);
   }

   FALCON_FUNC opengl_ShadeModel( VMachine *vm )
   {
      Item *i_mode = vm->param(0);
      if ( ( i_mode == 0 || !i_mode->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N" ) ) ;
         return;
      }

      GLenum  mode = (GLenum ) i_mode->forceInteger();
      ::glShadeModel(mode);
   }

   FALCON_FUNC opengl_Hint( VMachine *vm )
   {
      Item *i_target = vm->param(0);
      Item *i_mode = vm->param(1);
      if ( ( i_target == 0 || !i_target->isOrdinal() ) ||
           ( i_mode == 0 || !i_mode->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N" ) ) ;
         return;
      }
      
      GLenum  target = (GLenum ) i_target->forceInteger();
      GLenum  mode = (GLenum ) i_mode->forceInteger();
      ::glHint (target, mode);
   }
   
   FALCON_FUNC opengl_DepthFunc( VMachine *vm )
   {
      Item *i_func = vm->param(0);
      if ( ( i_func == 0 || !i_func->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N" ) ) ;
         return;
      }

      GLenum  func = (GLenum ) i_func->forceInteger();
      ::glDepthFunc(func);
   }

   FALCON_FUNC opengl_TexCoord2d( VMachine *vm )
   {
      Item *i_s = vm->param(0);
      Item *i_t = vm->param(1);
      if ( ( i_s == 0 || !i_s->isOrdinal() ) ||
           ( i_t == 0 || !i_t->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N" ) ) ;
         return;
      }
      
      GLdouble s = (GLdouble ) i_s->forceNumeric();
      GLdouble t = (GLdouble ) i_t->forceNumeric();
      ::glTexCoord2d (s, t);
   }

   FALCON_FUNC opengl_BindTexture( VMachine *vm )
   {
      Item *i_target = vm->param(0);
      Item *i_texture = vm->param(1);
      if ( ( i_target == 0 || !i_target->isOrdinal() ) ||
           ( i_texture == 0 || !i_texture->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N" ) ) ;
         return;
      }
      
      GLenum  target = (GLenum ) i_target->forceInteger();
      GLuint  texture = (GLuint ) i_texture->forceInteger();
      ::glBindTexture (target, texture);
   }

   FALCON_FUNC opengl_TexParameteri( VMachine *vm )
   {
      Item *i_target = vm->param(0);
      Item *i_pname = vm->param(1);
      Item *i_param = vm->param(2);
      if ( ( i_target == 0 || !i_target->isOrdinal() ) ||
           ( i_pname == 0 || !i_pname->isOrdinal() ) ||
           ( i_param == 0 || !i_param->isOrdinal() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N" ) ) ;
         return;
      }
      
      GLenum  target = (GLenum ) i_target->forceInteger();
      GLenum  pname = (GLenum ) i_pname->forceInteger();
      GLint param = (GLint ) i_param->forceInteger();
      ::glTexParameteri (target, pname, param);
   }

   FALCON_FUNC opengl_TexImage2D( VMachine *vm )
   {
      Item *i_target = vm->param(0);
      Item *i_level = vm->param(1);
      Item *i_internalformat = vm->param(2);
      Item *i_width = vm->param(3);
      Item *i_height = vm->param(4);
      Item *i_border = vm->param(5);
      Item *i_format = vm->param(6);
      Item *i_type = vm->param(7);
      Item *i_pixels = vm->param(8);
      if ( ( i_target == 0 || !i_target->isOrdinal() ) ||
           ( i_level == 0 || !i_level->isOrdinal() ) ||
           ( i_internalformat == 0 || !i_internalformat->isOrdinal() ) ||
           ( i_width == 0 || !i_width->isOrdinal() ) ||
           ( i_height == 0 || !i_height->isOrdinal() ) ||
           ( i_border == 0 || !i_border->isOrdinal() ) ||
           ( i_format == 0 || !i_format->isOrdinal() ) ||
           ( i_type == 0 || !i_type->isOrdinal() ) ||
           ( i_pixels == 0 || !i_pixels->isMemBuf() )
         )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "N,N,N,N,N,N,N,N,M" ) ) ;
         return;
      }
      
      GLenum  target = (GLenum ) i_target->forceInteger();
      GLint  level = (GLint ) i_level->forceInteger();
      GLint internalformat = (GLint ) i_internalformat->forceInteger();
      GLsizei width = (GLsizei) i_width->forceInteger();
      GLsizei height = (GLsizei) i_height->forceInteger();
      GLint  border = (GLint ) i_border->forceInteger();
      GLenum  format = (GLenum ) i_format->forceInteger();
      GLenum  type = (GLenum ) i_type->forceInteger();
      const GLvoid *pixels = (const GLvoid *) i_pixels->asMemBuf()->data();
      ::glTexImage2D (target, level, internalformat, width, height, border, format, type, pixels);
   }

}
}

/* end of gl_ext.cpp */
