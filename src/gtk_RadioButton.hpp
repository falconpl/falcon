#ifndef GTK_RADIOBUTTON_HPP
#define GTK_RADIOBUTTON_HPP

#include "modgtk.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::RadioButton
 */
class RadioButton
    :
    public Gtk::CoreGObject
{
public:

    RadioButton( const Falcon::CoreClass*, const GtkRadioButton* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_group_changed( VMARG );

    static void on_group_changed( GtkRadioButton*, gpointer );

    static FALCON_FUNC get_group( VMARG );

    static FALCON_FUNC set_group( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_RADIOBUTTON_HPP
