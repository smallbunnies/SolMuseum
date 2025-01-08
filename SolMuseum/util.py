from Solverz import Model, Var, Param, Eqn, Ode


def rename_mdl(m: Model, newname):

    # do rename
    # for example, replace m.ux by m.__dict__['ux_'+name]
    # to prevent the namespace pollution when there is gt_syn or st_syn
    ele_name_list = []
    new_dict = dict()
    for ele_name, value in m.__dict__.items():
        if isinstance(value, (Var, Param, Eqn, Ode)):
            new_dict[ele_name + '_' + newname] = value
            ele_name_list.append(ele_name)

    # dictionary keys not allowed to be changed during iteration
    for ele_name in ele_name_list:
        m.__dict__.pop(ele_name)
    m.__dict__.update(new_dict)

    return m
