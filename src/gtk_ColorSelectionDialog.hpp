#ifndef GTK_COLORSELECTIONDIALOG_HPP
#define GTK_COLORSELECTIONDIALOG_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::ColorSelectionDialog
 */
class ColorSelectionDialog
    :
    public Gtk::CoreGObject
{
public:

    ColorSelectionDialog( const Falcon::CoreClass*, const GtkColorSelectionDialog* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

#if GTK_MINOR_VERSION >= 14
    static FALCON_FUNC get_color_selection( VMARG );
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_COLORSELECTIONDIALOG_HPP
