/**
 *  \file gtk_Scrollbar.cpp
 */

#include "gtk_Scrollbar.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Scrollbar::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Scrollbar = mod->addClass( "GtkScrollbar", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkRange" ) );
    c_Scrollbar->getClassDef()->addInheritance( in );

}


/*#
    @class GtkScrollbar
    @brief Base class for GtkHScrollbar and GtkVScrollbar

    The GtkScrollbar widget is an abstract base class for GtkHScrollbar and
    GtkVScrollbar. It is not very useful in itself.

    The position of the thumb in a scrollbar is controlled by the scroll
    adjustments. See GtkAdjustment for the fields in an adjustment - for
    GtkScrollbar, the "value" field represents the position of the scrollbar,
    which must be between the "lower" field and "upper - page_size."
    The "page_size" field represents the size of the visible scrollable area.
    The "step_increment" and "page_increment" fields are used when the user
    asks to step down (using the small stepper arrows) or page down (using
    for example the PageDown key).
 */


} // Gtk
} // Falcon
