/**
 *  \file gtk_ActionGroup.cpp
 */

#include "gtk_ActionGroup.hpp"

#include "gtk_Action.hpp"
#include "gtk_Buildable.hpp"

/*#
    @beginmodule gtk
 */

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ActionGroup::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ActionGroup = mod->addClass( "GtkActionGroup", &ActionGroup::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_ActionGroup->getClassDef()->addInheritance( in );

    c_ActionGroup->setWKS( true );
    c_ActionGroup->getClassDef()->factory( &ActionGroup::factory );

    Gtk::MethodTab methods[] =
    {
    { "add_action_with_accel",      &ActionGroup::add_action_with_accel },
    // todo...
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ActionGroup, meth->name, meth->cb );

    Gtk::Buildable::clsInit( mod, c_ActionGroup );
}


ActionGroup::ActionGroup( const Falcon::CoreClass* gen, const GtkActionGroup* grp )
    :
    Gtk::CoreGObject( gen, (GObject*) grp )
{}


Falcon::CoreObject* ActionGroup::factory( const Falcon::CoreClass* gen, void* grp, bool )
{
    return new ActionGroup( gen, (GtkActionGroup*) grp );
}


/*#
    @class GtkActionGroup
    @brief A group of actions
    @param name the name of the action group.

    Actions are organised into groups. An action group is essentially a map from
    names to GtkAction objects.

    All actions that would make sense to use in a particular context should be
    in a single group. Multiple action groups may be used for a particular user
    interface. In fact, it is expected that most nontrivial applications wil
    make use of multiple groups. For example, in an application that can edit
    multiple documents, one group holding global actions (e.g. quit, about, new),
    and one group per document holding actions that act on that document (eg.
    save, cut/copy/paste, etc). Each window's menus would be constructed from a
    combination of two action groups.

    Accelerators are handled by the GTK+ accelerator map. All actions are assigned
    an accelerator path (which normally has the form <Actions>/group-name/action-name)
    and a shortcut is associated with this accelerator path. All menuitems and
    toolitems take on this accelerator path. The GTK+ accelerator map code makes
    sure that the correct shortcut is displayed next to the menu item.

    [...]
 */
FALCON_FUNC ActionGroup::init( VMARG )
{
    Item* i_nm = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_nm || !i_nm->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    if ( self->getObject() )
        return;
    AutoCString nm( i_nm->asString() );
    self->setObject( (GObject*) gtk_action_group_new( nm.c_str() ) );
}


/*#
    @method add_action_with_accel GtkActionGroup
    @brief Adds an action object to the action group and sets up the accelerator.
    @param action the action to add
    @param accelerator the accelerator for the action, in the format understood by GtkAccelerator.parse(), or "" for no accelerator, or Nil to use the stock accelerator.

    If accelerator is Nil, attempts to use the accelerator associated with the
    stock_id of the action.

    Accel paths are set to <Actions>/group-name/action-name.
 */
FALCON_FUNC ActionGroup::add_action_with_accel( VMARG )
{
    Item* i_act = vm->param( 0 );
    Item* i_accel = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_act || !( i_act->isObject() && IS_DERIVED( i_act, GtkAction ) )
        || !i_accel || !( i_accel->isString() || i_accel->isNil() ) )
        throw_inv_params( "GtkAction,[S]" );
#endif
    const gchar* acc = NULL;
    AutoCString accel;
    if ( !i_accel->isNil() )
    {
        accel.set( *i_accel->asString() );
        acc = accel.c_str();
    }
    gtk_action_group_add_action_with_accel( GET_ACTIONGROUP( vm->self() ),
                                            GET_ACTION( *i_act ),
                                            acc );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4 ts=4 sts=4:
// kate: replace-tabs on; shift-width 4;
