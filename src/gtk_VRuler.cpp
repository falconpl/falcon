/**
 *  \file gtk_VRuler.cpp
 */

#include "gtk_VRuler.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void VRuler::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Ruler = mod->addClass( "GtkVRuler", &VRuler::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkRuler" ) );
    c_Ruler->getClassDef()->addInheritance( in );

    /*
     *  implements GtkBuilable
     */
    
    /*
     *  implements GtkOrientable
     */

}


VRuler::VRuler( const Falcon::CoreClass* gen, const GtkVRuler* ruler )
    :
    Gtk::CoreGObject( gen, (GObject*) ruler )
{}


Falcon::CoreObject* VRuler::factory( const Falcon::CoreClass* gen, void* ruler, bool )
{
    return new VRuler( gen, (GtkVRuler*) ruler );
}


/*#
    @class GtkVRuler
    @brief A vertical ruler

    Note: This widget is considered too specialized/little-used for GTK+, and will
    in the future be moved to some other package. If your application needs this widget,
    feel free to use it, as the widget does work and is useful in some applications;
    it's just not of general interest. However, we are not accepting new features for
    the widget, and it will eventually move out of the GTK+ distribution.

    The VRuler widget is a widget arranged vertically creating a ruler that is utilized
    around other widgets such as a text widget. The ruler is used to show the location
    of the mouse on the window and to show the size of the window in specified units.
    The available units of measurement are GTK_PIXELS, GTK_INCHES and GTK_CENTIMETERS.
    GTK_PIXELS is the default.
 */
FALCON_FUNC VRuler::init( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GtkWidget* ruler = gtk_vruler_new();
    self->setGObject( (GObject*) ruler );
}


} // Gtk
} // Falcon
