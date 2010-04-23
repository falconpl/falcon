/**
 *  \file gtk_HRuler.cpp
 */

#include "gtk_HRuler.hpp"

#include "gtk_Buildable.hpp"
//#include "gtk_Orientable.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void HRuler::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_HRuler = mod->addClass( "GtkHRuler", &HRuler::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkRuler" ) );
    c_HRuler->getClassDef()->addInheritance( in );

#if GTK_MINOR_VERSION >= 12
    /*
     *  implements GtkBuildable
     */
    Gtk::Buildable::clsInit( mod, c_HRuler );
#endif

    /*
     *  implements GtkOrientable
     */
    //Gtk::Orientable::clsInit( mod, c_HRuler );

}


HRuler::HRuler( const Falcon::CoreClass* gen, const GtkHRuler* ruler )
    :
    Gtk::CoreGObject( gen, (GObject*) ruler )
{}


Falcon::CoreObject* HRuler::factory( const Falcon::CoreClass* gen, void* ruler, bool )
{
    return new HRuler( gen, (GtkHRuler*) ruler );
}


/*#
    @class GtkHRuler
    @brief A horizontal ruler

    Note: This widget is considered too specialized/little-used for GTK+, and will
    in the future be moved to some other package. If your application needs this widget,
    feel free to use it, as the widget does work and is useful in some applications;
    it's just not of general interest. However, we are not accepting new features for
    the widget, and it will eventually move out of the GTK+ distribution.

    The HRuler widget is a widget arranged horizontally creating a ruler that is
    utilized around other widgets such as a text widget. The ruler is used to show
    the location of the mouse on the window and to show the size of the window in
    specified units. The available units of measurement are GTK_PIXELS, GTK_INCHES
    and GTK_CENTIMETERS. GTK_PIXELS is the default.
 */
FALCON_FUNC HRuler::init( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GtkWidget* ruler = gtk_hruler_new();
    self->setGObject( (GObject*) ruler );
}


} // Gtk
} // Falcon
