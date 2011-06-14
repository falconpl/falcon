/**
 *  \file gtk_AspectFrame.cpp
 */

#include "gtk_AspectFrame.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void AspectFrame::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_AspectFrame = mod->addClass( "GtkAspectFrame", &AspectFrame::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkFrame" ) );
    c_AspectFrame->getClassDef()->addInheritance( in );

    c_AspectFrame->getClassDef()->factory( &AspectFrame::factory );

    mod->addClassMethod( c_AspectFrame, "set", &AspectFrame::set );
}


AspectFrame::AspectFrame( const Falcon::CoreClass* gen, const GtkAspectFrame* frame )
    :
    Gtk::CoreGObject( gen, (GObject*) frame )
{}

Falcon::CoreObject* AspectFrame::factory( const Falcon::CoreClass* gen, void* frame, bool )
{
    return new AspectFrame( gen, (GtkAspectFrame*) frame );
}


/*#
    @class GtkAspectFrame
    @brief A frame that constrains its child to a particular aspect ratio
    @optparam label Label text.
    @optparam xalign Horizontal alignment of the child within the allocation of the GtkAspectFrame. This ranges from 0.0 (left aligned) to 1.0 (right aligned)
    @optparam yalign Vertical alignment of the child within the allocation of the GtkAspectFrame. This ranges from 0.0 (left aligned) to 1.0 (right aligned)
    @optparam ratio The desired aspect ratio.
    @optparam obey_child If true, ratio is ignored, and the aspect ratio is taken from the requistion of the child.

    The GtkAspectFrame is useful when you want pack a widget so that it can resize
    but always retains the same aspect ratio. For instance, one might be drawing a
    small preview of a larger image. GtkAspectFrame derives from GtkFrame, so it can
    draw a label and a frame around the child. The frame will be "shrink-wrapped"
    to the size of the child.
 */
FALCON_FUNC AspectFrame::init( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S,N,N,N,B]" );

    const char* lbl = args.getCString( 0, false );
    gfloat xalign = args.getNumeric( 1, false );
    gfloat yalign = args.getNumeric( 2, false );
    gfloat ratio = args.getNumeric( 3, false );
    gboolean obey = args.getBoolean( 4, false );

    MYSELF;
    GtkWidget* wdt = gtk_aspect_frame_new(
        lbl ? lbl : "", xalign, yalign, ratio, obey );
    self->setObject( (GObject*) wdt );
}


/*#
    @method set GtkAspectFrame
    @brief Set parameters for an existing GtkAspectFrame.
    @param xalign Horizontal alignment of the child within the allocation of the GtkAspectFrame. This ranges from 0.0 (left aligned) to 1.0 (right aligned)
    @param yalign Vertical alignment of the child within the allocation of the GtkAspectFrame. This ranges from 0.0 (left aligned) to 1.0 (right aligned)
    @param ratio The desired aspect ratio.
    @param obey_child If true, ratio is ignored, and the aspect ratio is taken from the requistion of the child.
 */
FALCON_FUNC AspectFrame::set( VMARG )
{
    Gtk::ArgCheck0 args( vm, "N,N,N,B" );

    gfloat xalign = args.getNumeric( 0 );
    gfloat yalign = args.getNumeric( 1 );
    gfloat ratio = args.getNumeric( 2 );
    gboolean obey = args.getBoolean( 3 );

    MYSELF;
    GET_OBJ( self );
    gtk_aspect_frame_set( (GtkAspectFrame*)_obj, xalign, yalign, ratio, obey );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
